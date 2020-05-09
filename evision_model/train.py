# -*- coding: utf-8 -*-
"""
A driver for training models

code by jiafeng5513

"""
import argparse
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import sys
import csv

import models
import custom_transforms

"""命令行参数"""
parser = argparse.ArgumentParser(description='EvisionNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
"""程序参数"""
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='数据格式, stacked:在彩色通道堆叠;sequential:在width方向;连接')
parser.add_argument('data', metavar='path to KittiRaw_formatted', help='预处理后的数据集路径')
parser.add_argument('--with-gt', action='store_true', help='验证时是否使用GT,若要使用,在数据准备时需要使用--with-depth')
parser.add_argument('--pretrained-depthnet', dest='pretrained_depth', default=None, metavar='PATH', help='预训练DepthNet的路径')
parser.add_argument('--pretrained-motionnet', dest='pretrained_motion', default=None, metavar='PATH', help='预训练MotionNet的路径')
"""超参数"""
parser.add_argument('--sequence-length', type=int, metavar='N', help='每个训练样本由几帧构成', default=3)
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,metavar='LR', help='学习率')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--img_height', default=128, type=int, help='Input frame height.')
parser.add_argument('--img_width', default=416, type=int, help='Input frame width.')
parser.add_argument('--seed', default=8964, type=int, help='随机数种子')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='SGD的动量,Adam的alpha')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='Adam的beta')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
"""损失函数权重"""
parser.add_argument('--motion_smoothing_weight', default=1e-3, type=float, help='场景平移场平滑损失权重')
parser.add_argument('--depth_smoothing_weight', default=1e-2, type=float, help='深度平滑损失权重')
parser.add_argument('--depth_consistency_loss_weight', default=1e-2, type=float, help='深度一致性损失权重')
parser.add_argument('--rgb_weight', default=0.85, type=float, help='rgb惩罚权重')
parser.add_argument('--ssim_weight', default=1.5, type=float, help='结构相似性平衡惩罚权重')
parser.add_argument('--rotation_consistency_weight', default=1e-3, type=float, help='旋转一致性损失权重')
parser.add_argument('--translation_consistency_weight', default=1e-2, type=float, help='平移一致性损失权重')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)

    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['sequence_length'] = 'seq'
    keys_with_prefix['rotation_mode'] = 'rot_'
    keys_with_prefix['padding_mode'] = 'padding_'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'
    keys_with_prefix['photo_loss_weight'] = 'p'
    keys_with_prefix['mask_loss_weight'] = 'm'
    keys_with_prefix['smooth_loss_weight'] = 's'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    if args.intri_pred:
        folder_string.append('calib')
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H%M")
    return save_path / timestamp


def main():
    global best_error, n_iter, device
    args = parser.parse_args()

    """======step 1 : 根据使用的数据类型加载相应的数据流水线======"""
    if args.dataset_format == 'stacked':
        from datasets.stacked_sequence_folders import SequenceFolder
    elif args.dataset_format == 'sequential':
        from datasets.sequence_folders import SequenceFolder
    """======step 2 : 准备存储目录======"""
    save_path = save_path_formatter(args, parser)
    if sys.platform is 'win32':
        args.save_path = '.\checkpoints' / save_path
    else:  # linux
        args.save_path = 'checkpoints' / save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    """======step 3 : 指定随机数种子以便于实验复现======"""
    torch.manual_seed(args.seed)

    if args.evaluate:
        args.epochs = 0

    # tb_writer = SummaryWriter(args.save_path)

    """=========step 4 : 数据准备=========="""
    # 数据扩增
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([custom_transforms.RandomHorizontalFlip(),
                                                 custom_transforms.RandomScaleCrop(),
                                                 custom_transforms.ArrayToTensor(), normalize])
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching data from '{}'".format(args.data))
    train_set = SequenceFolder(args.data, transform=train_transform, seed=args.seed, train=True,
                               sequence_length=args.sequence_length)

    # if no Groundtruth is avalaible,
    # Validation set is the same type as training set to measure photometric loss from warping
    if args.with_gt:
        from datasets.validation_folders import ValidationSet
        val_set = ValidationSet(args.data, transform=valid_transform)
    else:
        val_set = SequenceFolder(args.data, transform=valid_transform, seed=args.seed, train=False, sequence_length=args.sequence_length)

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    """========= step 5 : 加载模型 =========="""
    print("=> creating models")

    depth_net = models.DepthNet().to(device)
    motion_net = models.MotionNet().to(device)

    if args.pretrained_depth:
        print("=> using pre-trained weights for DepthNet")
        weights = torch.load(args.pretrained_depth)
        depth_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        depth_net.init_weights()

    if args.pretrained_motion:
        print("=> using pre-trained weights for MotionNet")
        weights = torch.load(args.pretrained_motion)
        motion_net.load_state_dict(weights['state_dict'])
    else:
         motion_net.init_weights()

    cudnn.benchmark = True
    depth_net = torch.nn.DataParallel(depth_net)
    motion_net = torch.nn.DataParallel(motion_net)

    """========= step 6 : 设置求解器 =========="""
    print('=> setting adam solver')

    optim_params = [{'params': depth_net.parameters(), 'lr': args.lr},
                    {'params': motion_net.parameters(), 'lr': args.lr}]

    optimizer = torch.optim.Adam(optim_params, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)

    """========= step 7 : """
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

        # Up to you to chose the most relevant error to measure your models's performance,
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
        disparities = disp_net(tgt_img)  #
        # depth = [1 / disp for disp in disparities]
        depth = disparities
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
