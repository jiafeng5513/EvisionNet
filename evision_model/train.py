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
from path import Path
import datetime
from tqdm import tqdm
import time
import shutil
from tensorboardX import SummaryWriter
import models
import custom_transforms
from DataFlow.validation_folders import ValidationSet
from LossFunction import LossFactory
from evision_utils import tensor2array,AverageMeter


"""命令行参数"""
parser = argparse.ArgumentParser(description='EvisionNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
"""程序参数"""
parser.add_argument('data', metavar='path to KittiRaw_formatted', help='预处理后的数据集路径')
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='数据格式, stacked:在彩色通道堆叠;sequential:在width方向;连接')
parser.add_argument('--with-gt', action='store_true', help='验证时是否使用GT,若要使用,在数据准备时需要使用--with-depth')
parser.add_argument('--pretrained-depthnet', dest='pretrained_depth', default=None, metavar='PATH',
                    help='预训练DepthNet的路径')
parser.add_argument('--pretrained-motionnet', dest='pretrained_motion', default=None, metavar='PATH',
                    help='预训练MotionNet的路径')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='数据加载线程数')
"""超参数"""
parser.add_argument('--SEQ_LENGTH', default=3, type=int, metavar='N', help='每个训练样本由几帧构成')
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR', help='学习率')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--epoch-size', default=3000, type=int, metavar='N', help='手动设置每个epoch的样本数量')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='训练循环的次数')
parser.add_argument('--img_height', default=128, type=int, help='Input frame height.')
parser.add_argument('--img_width', default=416, type=int, help='Input frame width.')
parser.add_argument('--seed', default=8964, type=int, help='随机数种子')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='SGD的动量,Adam的alpha')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='Adam的beta')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--intri_pred', dest='intri_pred', action='store_true', help='open=predict intri,close = gt intri')
"""损失函数权重"""
parser.add_argument('--motion_smoothing_weight', default=1e-3, type=float, help='场景平移场平滑损失权重')
parser.add_argument('--depth_smoothing_weight', default=1e-2, type=float, help='深度平滑损失权重')
parser.add_argument('--depth_consistency_loss_weight', default=1e-2, type=float, help='深度一致性损失权重')
parser.add_argument('--rgb_weight', default=0.85, type=float, help='rgb惩罚权重')
parser.add_argument('--ssim_weight', default=1.5, type=float, help='结构相似性平衡惩罚权重')
parser.add_argument('--rotation_consistency_weight', default=1e-3, type=float, help='旋转一致性损失权重')
parser.add_argument('--translation_consistency_weight', default=1e-2, type=float, help='平移一致性损失权重')

parser.add_argument('--log-output', action='store_true', help='开启后,验证期间dispnet的输出和重投影图片会被保存')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('-f', '--training-output-freq', type=int, help='训练期间输出dispnet和重投影图片的频率,设为0则不输出',
                    metavar='N', default=0)
"""   全局变量   """
best_error = -1  # 用于识别当前最佳的模型状态
n_iter = 0  # 训练的次数
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 计算设备


def main():
    global best_error, n_iter, device
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(True)  # 启动梯度侦测,用于查找梯度终断
    """====== step 1 : 根据使用的数据类型加载相应的数据流水线  ======"""
    if args.dataset_format == 'stacked':
        from DataFlow.stacked_sequence_folders import SequenceFolder
    elif args.dataset_format == 'sequential':
        from DataFlow.sequence_folders import SequenceFolder
    """====== step 2 : 准备存储目录 ======"""
    save_path = save_path_formatter(args, parser)
    if sys.platform is 'win32':
        args.save_path = '.\checkpoints' / save_path
    else:  # linux
        args.save_path = 'checkpoints' / save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    tb_writer = SummaryWriter(args.save_path)  # tensorboardx writer
    """====== step 3 : 指定随机数种子以便于实验复现 ======"""
    torch.manual_seed(args.seed)

    """========= step 4 : 数据准备 =========="""
    # 数据扩增
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([custom_transforms.RandomHorizontalFlip(),
                                                 custom_transforms.RandomScaleCrop(),
                                                 custom_transforms.ArrayToTensor(), normalize])
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
    # 训练集
    print("=> fetching data from '{}'".format(args.data))
    train_set = SequenceFolder(args.data, transform=train_transform, seed=args.seed, train=True,
                               sequence_length=args.SEQ_LENGTH)
    # 验证集
    val_set = ValidationSet(args.data, transform=valid_transform)

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    """========= step 5 : 加载模型 =========="""
    print("=> creating models")

    depth_net = models.DepthNet().to(device)
    motion_net = models.MotionNet(intrinsic_pred=args.intri_pred).to(device)

    if args.pretrained_depth:
        print("=> using pre-trained weights for DepthNet")
        weights = torch.load(args.pretrained_depth)
        depth_net.load_state_dict(weights['state_dict'], strict=False)


    if args.pretrained_motion:
        print("=> using pre-trained weights for MotionNet")
        weights = torch.load(args.pretrained_motion)
        motion_net.load_state_dict(weights['state_dict'])


    cudnn.benchmark = True
    depth_net = torch.nn.DataParallel(depth_net)
    motion_net = torch.nn.DataParallel(motion_net)

    """========= step 6 : 设置求解器 =========="""
    print('=> setting adam solver')

    optim_params = [{'params': depth_net.parameters(), 'lr': args.lr},
                    {'params': motion_net.parameters(), 'lr': args.lr}]

    optimizer = torch.optim.Adam(optim_params, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)
    """====== step 7 : 初始化损失函数计算器======="""
    total_loss_calculator = LossFactory(SEQ_LENGTH=args.SEQ_LENGTH,
                                        rgb_weight=args.rgb_weight,
                                        depth_smoothing_weight=args.depth_smoothing_weight,
                                        ssim_weight=args.ssim_weight,
                                        motion_smoothing_weight=args.motion_smoothing_weight,
                                        rotation_consistency_weight=args.rotation_consistency_weight,
                                        translation_consistency_weight=args.translation_consistency_weight,
                                        depth_consistency_loss_weight=args.depth_consistency_loss_weight)
    """========= step 8 : 训练循环 =========="""
    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)  # 如果不指定epoch_size,那么每一个epoch就把全部的训练数据过一遍
    for epoch in range(args.epochs):
        tqdm.write("\n===========TRAIN EPOCH [{}/{}]===========".format(epoch + 1, args.epochs))
        """====== step 8.1 : 训练一个epoch ======"""
        train_loss = train(args, train_loader, depth_net, motion_net, optimizer,
                           args.epoch_size, total_loss_calculator,tb_writer)
        tqdm.write('* Avg Loss : {:.3f}'.format(train_loss))
        """======= step 8.2 : 验证 ========"""
        # 验证时要输出 : 深度指标abs_diff, abs_rel, sq_rel, a1, a2, a3
        errors, error_names = validate_with_gt(args, val_loader, depth_net, motion_net, epoch)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        tqdm.write(error_string)
        # TODO:输出验证集上的轨迹指标

        for error, name in zip(errors, error_names):
            tb_writer.add_scalar(name, error, epoch)

        """======= step 8.3 : 保存验证效果最佳的模型状态 =========="""
        decisive_error = errors[1]  # 选取abs_real作为关键评价指标,注意论文上我们以a3为关键指标
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(args.save_path, {'epoch': epoch + 1, 'state_dict': depth_net.module.state_dict()},
                        {'epoch': epoch + 1, 'state_dict': motion_net.module.state_dict()}, is_best)
    pass  # end of main


# 训练一个epoch
def train(args, train_loader, depth_net, motion_net, optimizer, epoch_size, loss_calculator, tb_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)

    """============ 1. 把模型切换到训练模式 =========="""
    depth_net.train()
    motion_net.train()

    end = time.time()
    """============ 2. 准备好epoch内的训练输出模式 =========="""
    train_pbar = tqdm(total=min(len(train_loader), args.epoch_size),
                      bar_format='{desc} {percentage:3.0f}%|{bar}| {postfix}')
    train_pbar.set_description('Train: Total Loss=#.####(#.####)')
    train_pbar.set_postfix_str('<TIME: op=#.###(#.###) data=#.###(#.###)>')
    """============ 3. 开始训练这一个epoch的数据============"""
    for i, (imgs,  gt_intrinsics) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0
        log_output = args.training_output_freq > 0 and n_iter % args.training_output_freq == 0

        """======3.1 计时======"""
        data_time.update(time.time() - end)
        """======3.2 传输数据到计算设备======"""
        #imgs = imgs.to(device)  # list of [B,3,h,w], list length = SEQ_LENGTH
        imgs = [img.to(device) for img in imgs]
        gt_intrinsics = gt_intrinsics.to(device)  # [3,3]
        """======3.3 计算网络模型的输出======"""
        depth_pred_list = []
        for item in imgs:
            depth_pred_list.append(depth_net(item))  # 深度预测
        rot_pred_list = []
        inv_rot_pred_list = []
        trans_pred_list = []
        inv_trans_pred_list = []
        trans_res_pred_list = []
        inv_trans_res_pred_list = []
        intrinsic_pred = torch.zeros(3, 3).repeat(4, 1, 1).to(device)
        for j in range(args.SEQ_LENGTH - 1):
            a = j
            b = j + 1
            image_a = imgs[a]
            image_b = imgs[b]

            if args.intri_pred:
                # 输入的图像是二连帧,在第二个通道上堆叠[B,6,H,W]
                (rotation, translation, residual_translation, pred_intrinsic) = motion_net(
                    torch.cat((image_a, image_b), dim=1))
                rot_pred_list.append(rotation)
                trans_pred_list.append(translation)
                trans_res_pred_list.append(residual_translation)
                (inv_rotation, inv_translation, inv_residual_translation, inv_pred_intrinsic) = motion_net(
                    torch.cat((image_b, image_a), dim=1))
                inv_rot_pred_list.append(inv_rotation)
                inv_trans_pred_list.append(inv_translation)
                inv_trans_res_pred_list.append(inv_residual_translation)
                intrinsic_pred += (inv_pred_intrinsic + pred_intrinsic) * 0.5
            else:
                (rotation, translation, residual_translation) = motion_net(
                    torch.cat((image_a, image_b), dim=1))
                rot_pred_list.append(rotation)
                trans_pred_list.append(translation)
                trans_res_pred_list.append(residual_translation)
                (inv_rotation, inv_translation, inv_residual_translation) = motion_net(
                    torch.cat((image_b, image_a), dim=1))
                inv_rot_pred_list.append(inv_rotation)
                inv_trans_pred_list.append(inv_translation)
                inv_trans_res_pred_list.append(inv_residual_translation)
        intrinsic_pred = intrinsic_pred / (args.SEQ_LENGTH - 1)
        """======3.4 计算损失======"""
        if args.intri_pred:
            # last None for obj_mask,we turn obj_mask off
            loss_calculator.SolveTotalLoss(imgs, depth_pred_list, trans_pred_list, trans_res_pred_list, rot_pred_list,
                                           inv_trans_pred_list, inv_trans_res_pred_list, inv_rot_pred_list,
                                           intrinsic_pred, None)
        else:
            loss_calculator.getTotalLoss(imgs, depth_pred_list, trans_pred_list, trans_res_pred_list, rot_pred_list,
                                         inv_trans_pred_list, inv_trans_res_pred_list, inv_rot_pred_list,
                                         gt_intrinsics, None)
        total_loss = loss_calculator.getTotalLoss()
        # record loss and EPE
        losses.update(total_loss.item(), args.batch_size)

        if log_losses:
            tb_writer.add_scalar('Depth Consistency Loss', loss_calculator.getDepthConsistencyLoss(), n_iter)
            tb_writer.add_scalar('Depth Smoothing Loss', loss_calculator.getDepthSmoothingLoss(), n_iter)
            tb_writer.add_scalar('Motion Smoothing Loss', loss_calculator.getMotionSmoothingLoss(), n_iter)
            tb_writer.add_scalar('Rgb Penalty', loss_calculator.getRgbPenalty(), n_iter)
            tb_writer.add_scalar('SSIM Penalty', loss_calculator.getSsimPenalty(), n_iter)
            tb_writer.add_scalar('Rot Loss', loss_calculator.getRotLoss(), n_iter)
            tb_writer.add_scalar('Trans Loss', loss_calculator.getTransLoss(), n_iter)
            tb_writer.add_scalar('Total loss', total_loss, n_iter)

        """======3.5 梯度计算和更新======"""
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

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


# 验证
@torch.no_grad()
def validate_with_gt(args, val_loader, depth_net, motion_net, epoch, tb_writer, sample_nb_to_log=3):
    global device
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    log_outputs = sample_nb_to_log > 0
    # 切换到验证模式
    depth_net.eval()
    motion_net.eval()

    end = time.time()

    validate_pbar = tqdm(total=len(val_loader), bar_format='{desc} {percentage:3.0f}%|{bar}| {postfix}')
    validate_pbar.set_description('valid: Abs Error {:.4f} ({:.4f})'.format(0, 0))
    validate_pbar.set_postfix_str('<Time {} ({})>'.format(0, 0))

    for i, (tgt_img, depth) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        depth = depth.to(device)

        # compute output
        with torch.no_grad():
            output_depth = depth_net(tgt_img)

        if log_outputs and i < sample_nb_to_log:
            if epoch == 0:
                tb_writer.add_image('Val Input Image/{}'.format(i), tensor2array(tgt_img[0]), 0)
                tb_writer.add_image('GT Depth Map/{}'.format(i), tensor2array(depth[0], max_value=None), epoch)
            tb_writer.add_image('Pred Depth Map/{}'.format(i), tensor2array(output_depth[0], max_value=None), epoch)

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


# 深度指标
@torch.no_grad()
def compute_errors(gt, pred, crop=True):
    """
    计算深度指标
    Args:
        gt:    ground truth [B,128,416]
        pred:  prediction   [B,1,128,416]
        crop:
    Returns:
        abs_diff, abs_rel, sq_rel, a1, a2, a3
    """
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size = gt.size(0)

    mask_1 = gt > 0
    mask_2 = gt <= 80
    mask = mask_1 * mask_2
    pred_masked = pred.squeeze() * mask

    for current_gt, current_pred in zip(gt, pred_masked):
        # 对当前gt做均值方差归一化
        valid_gt = ((current_gt - current_gt.min()) / (current_gt.max() - current_gt.min())).clamp(1e-3, 1)
        valid_pred = ((current_pred - current_pred.min()) / (current_pred.max() - current_pred.min())).clamp(1e-3, 1)

        valid_pred = valid_pred * torch.median(valid_gt) / torch.median(valid_pred)  # 不明白

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred) ** 2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]


def save_path_formatter(args, parser):
    # TODO:超参数的名字需要更新
    def is_default(key, value):
        return value == parser.get_default(key)

    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['SEQ_LENGTH'] = 'seq'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    if args.intri_pred:
        folder_string.append('calib')
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H%M")
    return save_path / timestamp


def save_checkpoint(save_path, depth_net_state, motion_net_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['depth_net', 'motion_net']
    states = [depth_net_state, motion_net_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path / '{}_{}'.format(prefix, filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path / '{}_{}'.format(prefix, filename),
                            save_path / '{}_model_best.pth.tar'.format(prefix))


if __name__ == '__main__':
    main()
    pass