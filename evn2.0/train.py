import argparse
import time
import csv
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models, custom_transforms
from my_utils import tensor2array, save_checkpoint, save_path_formatter, log_output_tensorboard, intrinsics_pred_decode,AverageMeter
from loss_functions import photometric_reconstruction_loss, explainability_loss, smooth_loss, compute_errors
from tensorboardX import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser(description='EvisionNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

"""Program Initialization parameters"""
parser.add_argument('data', metavar='/home/RAID1/DataSet/KITTI/KittiRaw_formatted/', help='预处理后的数据集路径')
parser.add_argument('--dataset-format', default='sequential', metavar='STR', help='数据格式, stacked:连帧;sequential:单帧序列')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='数据加载线程数')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='预训练Disp-Net的路径')
parser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH',
                    help='预训练pose-net的路径')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='保存每个epoch的训练和验证情况的csv文件名')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='保存训练期间每次梯度下降后的情况的csv')

"""Init-parameters of training"""
parser.add_argument('--with-gt', action='store_true', help='验证时是否使用GT,若要使用,在数据准备时需要使用--with-depth')
parser.add_argument('--sequence-length', type=int, metavar='N', help='每个训练样本由几帧构成', default=3)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='旋转表示方法: euler:欧拉角(yaw,pitch,roll);quaternion:四元数 (last 3 coefficients)')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='重投影期间的延拓模式, 这会影响重投影误差的计算.'
                         ' zeros: will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')

"""Hyper-parameters of training"""
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='训练多少epoch')
parser.add_argument('--epoch-size', default=3000, type=int, metavar='N', help='手动设置每个epoch的样本数量,如果不设置将会根据数据集的情况定值')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR', help='学习率')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='SGD的动量,Adam的alpha')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='Adam的beta')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--seed', default=0, type=int, help='随机种子')
parser.add_argument('-p', '--photo-loss-weight', type=float, help='一致性损失的权重', metavar='W', default=1)
parser.add_argument('-m', '--mask-loss-weight', type=float, help='mask损失的权重', metavar='W', default=0.2)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='视差平滑损失的权重', metavar='W', default=0.1)
parser.add_argument('--intri_pred', dest='intri_pred', action='store_true', help='open=predict intri,close = gt intri')

"""Program Behavior Parameters"""
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='打开这个将在验证集上评估模型,and skip training')
parser.add_argument('--log-output', action='store_true', help='开启后,验证期间dispnet的输出和重投影图片会被保存')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('-f', '--training-output-freq', type=int, help='训练期间输出dispnet和重投影图片的频率,设为0则不输出', metavar='N',
                    default=0)

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    global best_error, n_iter, device
    args = parser.parse_args()
    if args.dataset_format == 'stacked':
        from Pytorch_version.datasets.stacked_sequence_folders import SequenceFolder
    elif args.dataset_format == 'sequential':
        from Pytorch_version.datasets.sequence_folders import SequenceFolder
    save_path = save_path_formatter(args, parser)
    if sys.platform is 'win32':
        args.save_path = '.\checkpoints' / save_path
    else:  # linux
        args.save_path = 'checkpoints' / save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)
    if args.evaluate:
        args.epochs = 0

    tb_writer = SummaryWriter(args.save_path)
    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length
    )

    # if no Groundtruth is avalaible,
    # Validation set is the same type as training set to measure photometric loss from warping
    if args.with_gt:
        from Pytorch_version.datasets.validation_folders import ValidationSet
        val_set = ValidationSet(
            args.data,
            transform=valid_transform
        )
    else:
        val_set = SequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
        )
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    disp_net = models.DispNetS().to(device)
    output_exp = args.mask_loss_weight > 0
    if not output_exp:
        print("=> no mask loss, PoseExpnet will only output pose")
    pose_exp_net = models.PoseExpNet(nb_ref_imgs=args.sequence_length - 1, output_exp=args.mask_loss_weight > 0).to(
        device)

    if args.pretrained_exp_pose:
        print("=> using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_exp_pose)
        pose_exp_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        pose_exp_net.init_weights()

    if args.pretrained_disp:
        print("=> using pre-trained weights for Dispnet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'])
    else:
        disp_net.init_weights()

    cudnn.benchmark = True
    disp_net = torch.nn.DataParallel(disp_net)
    pose_exp_net = torch.nn.DataParallel(pose_exp_net)

    print('=> setting adam solver')

    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_exp_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)

    with open(args.save_path / args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path / args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'explainability_loss', 'smooth_loss'])

    if args.pretrained_disp or args.evaluate:
        if args.with_gt:
            errors, error_names = validate_with_gt(args, val_loader, disp_net, 0, tb_writer)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_exp_net, 0, tb_writer)
        for error, name in zip(errors, error_names):
            tb_writer.add_scalar(name, error, 0)
        error_string = ', '.join(
            '{} : {:.3f}'.format(name, error) for name, error in zip(error_names[2:9], errors[2:9]))
        tqdm.write(error_string)

    for epoch in range(args.epochs):
        tqdm.write("\n===========TRAIN EPOCH [{}/{}]===========".format(epoch + 1, args.epochs))
        train_loss = train(args, train_loader, disp_net, pose_exp_net, optimizer, args.epoch_size, tb_writer)
        tqdm.write('* Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        if args.with_gt:
            errors, error_names = validate_with_gt(args, val_loader, disp_net, epoch, tb_writer)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_exp_net, epoch, tb_writer)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        tqdm.write(error_string)

        for error, name in zip(errors, error_names):
            tb_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance,
        # careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_exp_net.module.state_dict()
            },
            is_best)

        with open(args.save_path / args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])


def train(args, train_loader, disp_net, pose_exp_net, optimizer, epoch_size, tb_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight

    # switch to train mode
    disp_net.train()
    pose_exp_net.train()

    end = time.time()

    train_pbar = tqdm(total=min(len(train_loader), args.epoch_size),
                      bar_format='{desc} {percentage:3.0f}%|{bar}| {postfix}')
    train_pbar.set_description('Train: Total Loss=#.####(#.####)')
    train_pbar.set_postfix_str('<TIME: op=#.###(#.###) data=#.###(#.###)>')

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0
        log_output = args.training_output_freq > 0 and n_iter % args.training_output_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        # compute output
        disparities = disp_net(tgt_img)
        depth = [1 / disp for disp in disparities]
        explainability_mask, pose, intrinsics_pred = pose_exp_net(tgt_img, ref_imgs)

        if args.intri_pred:
            # construct intrinsics[4,3,3] with intrinsics_pred[4,4]
            tmin = intrinsics_pred_decode(intrinsics_pred).to(device)
            loss_1, warped, diff = photometric_reconstruction_loss(tgt_img, ref_imgs, tmin,
                                                                   depth, explainability_mask, pose,
                                                                   args.rotation_mode, args.padding_mode)
        else:
            loss_1, warped, diff = photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics,
                                                                   depth, explainability_mask, pose,
                                                                   args.rotation_mode, args.padding_mode)
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask)
        else:
            loss_2 = 0
        loss_3 = smooth_loss(depth)

        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3

        if loss < 0.0005:
            abc = 0
        if log_losses:
            tb_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            if w2 > 0:
                tb_writer.add_scalar('explanability_loss', loss_2.item(), n_iter)
            tb_writer.add_scalar('disparity_smoothness_loss', loss_3.item(), n_iter)
            tb_writer.add_scalar('total_loss', loss.item(), n_iter)

        if log_output:
            tb_writer.add_image('train Input', tensor2array(tgt_img[0]), n_iter)
            for k, scaled_maps in enumerate(zip(depth, disparities, warped, diff, explainability_mask)):
                log_output_tensorboard(tb_writer, "train", 0, k, n_iter, *scaled_maps)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path / args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item() if w2 > 0 else 0, loss_3.item()])
        train_pbar.clear()
        train_pbar.update(1)
        train_pbar.set_description('Train: Total Loss={}'.format(losses))
        train_pbar.set_postfix_str('<TIME: op={} data={}>'.format(batch_time, data_time))
        if i >= epoch_size - 1:
            break

        n_iter += 1
    train_pbar.close()
    time.sleep(1)
    return losses.avg[0]


@torch.no_grad()
def validate_without_gt(args, val_loader, disp_net, pose_exp_net, epoch, tb_writer, sample_nb_to_log=3):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=3, precision=4)
    log_outputs = sample_nb_to_log > 0
    w1, w2, w3 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight
    poses = np.zeros(((len(val_loader) - 1) * args.batch_size * (args.sequence_length - 1), 6))
    disp_values = np.zeros(((len(val_loader) - 1) * args.batch_size * 3))

    # switch to evaluate mode
    disp_net.eval()
    pose_exp_net.eval()

    end = time.time()

    validate_pbar = tqdm(total=len(val_loader), bar_format='{desc} {percentage:3.0f}%|{bar}| {postfix}')
    validate_pbar.set_description('valid: Loss *.**** *.*****.****(*.**** *.**** *.****)')
    validate_pbar.set_postfix_str('<Time *.***(*.***)>')

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        # compute output
        disp = disp_net(tgt_img)
        depth = 1 / disp

        explainability_mask, pose, intrinsics_pred = pose_exp_net(tgt_img, ref_imgs)
        if args.intri_pred:
            # construct intrinsics[4,3,3] with intrinsics_pred[4,4]
            tmin = intrinsics_pred_decode(intrinsics_pred)
            loss_1, warped, diff = photometric_reconstruction_loss(tgt_img, ref_imgs,
                                                                   tmin, depth,
                                                                   explainability_mask, pose,
                                                                   args.rotation_mode, args.padding_mode)
        else:
            loss_1, warped, diff = photometric_reconstruction_loss(tgt_img, ref_imgs,
                                                                   intrinsics, depth,
                                                                   explainability_mask, pose,
                                                                   args.rotation_mode, args.padding_mode)
        loss_1 = loss_1.item()
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask).item()
        else:
            loss_2 = 0
        loss_3 = smooth_loss(depth).item()

        if log_outputs and i < sample_nb_to_log - 1:  # log first output of first batches
            if epoch == 0:
                for j, ref in enumerate(ref_imgs):
                    tb_writer.add_image('val Input {}/{}'.format(j, i), tensor2array(tgt_img[0]), 0)
                    tb_writer.add_image('val Input {}/{}'.format(j, i), tensor2array(ref[0]), 1)

            log_output_tensorboard(tb_writer, 'val', i, '', epoch, 1. / disp, disp, warped, diff, explainability_mask)

        if log_outputs and i < len(val_loader) - 1:
            step = args.batch_size * (args.sequence_length - 1)
            poses[i * step:(i + 1) * step] = pose.cpu().view(-1, 6).numpy()
            step = args.batch_size * 3
            disp_unraveled = disp.cpu().view(args.batch_size, -1)
            disp_values[i * step:(i + 1) * step] = torch.cat([disp_unraveled.min(-1)[0],
                                                              disp_unraveled.median(-1)[0],
                                                              disp_unraveled.max(-1)[0]]).numpy()

        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3
        losses.update([loss, loss_1, loss_2])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        validate_pbar.clear()
        validate_pbar.update(1)
        validate_pbar.set_description('valid: Loss {}'.format(losses))
        validate_pbar.set_postfix_str('<Time {}>'.format(batch_time))
    validate_pbar.close()

    if log_outputs:
        prefix = 'valid poses'
        coeffs_names = ['tx', 'ty', 'tz']
        if args.rotation_mode == 'euler':
            coeffs_names.extend(['rx', 'ry', 'rz'])
        elif args.rotation_mode == 'quat':
            coeffs_names.extend(['qx', 'qy', 'qz'])
        for i in range(poses.shape[1]):
            tb_writer.add_histogram('{} {}'.format(prefix, coeffs_names[i]), poses[:, i], epoch)
        tb_writer.add_histogram('disp_values', disp_values, epoch)
        time.sleep(0.2)
    else:
        time.sleep(1)
    return losses.avg, ['Total loss', 'Photo loss', 'Exp loss']


@torch.no_grad()
def validate_with_gt(args, val_loader, disp_net, epoch, tb_writer, sample_nb_to_log=3):
    global device
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    log_outputs = sample_nb_to_log > 0

    # switch to evaluate mode
    disp_net.eval()

    end = time.time()

    validate_pbar = tqdm(total=len(val_loader), bar_format='{desc} {percentage:3.0f}%|{bar}| {postfix}')
    validate_pbar.set_description('valid: Abs Error {:.4f} ({:.4f})'.format(0, 0))
    validate_pbar.set_postfix_str('<Time {}>'.format(0))

    for i, (tgt_img, depth) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        depth = depth.to(device)

        # compute output
        output_disp = disp_net(tgt_img)
        output_depth = 1 / output_disp[:, 0]

        if log_outputs and i < sample_nb_to_log:
            if epoch == 0:
                tb_writer.add_image('val Input/{}'.format(i), tensor2array(tgt_img[0]), 0)
                depth_to_show = depth[0]
                tb_writer.add_image('val target Depth Normalized/{}'.format(i),
                                    tensor2array(depth_to_show, max_value=None),
                                    epoch)
                depth_to_show[depth_to_show == 0] = 1000
                disp_to_show = (1 / depth_to_show).clamp(0, 10)
                tb_writer.add_image('val target Disparity Normalized/{}'.format(i),
                                    tensor2array(disp_to_show, max_value=None, colormap='magma'),
                                    epoch)

            tb_writer.add_image('val Dispnet Output Normalized/{}'.format(i),
                                tensor2array(output_disp[0], max_value=None, colormap='magma'),
                                epoch)
            tb_writer.add_image('val Depth Output Normalized/{}'.format(i),
                                tensor2array(output_depth[0], max_value=None),
                                epoch)

        errors.update(compute_errors(depth, output_depth))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        validate_pbar.update(1)
        validate_pbar.set_description('valid: Abs Error {:.4f} ({:.4f})'.format(errors.val[0], errors.avg[0]))
        validate_pbar.set_postfix_str('<Time {}>'.format(batch_time))

    validate_pbar.close()
    time.sleep(1)
    return errors.avg, error_names


if __name__ == '__main__':
    main()
