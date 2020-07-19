import torch
import tensorflow.compat.v1 as tf
import numpy as np
import torch.nn as nn
import time


# tf模板
def tensorflow_test_template():
    x = tf.placeholder(tf.float32, shape=[4, 128, 416, 1])
    # y = function_to_be_test(x)

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    x_vars = np.random.rand(4, 128, 416, 1)

    # r = sess.run(y, feed_dict={x: x_vars})
    # print(y.shape)
    # print(r)
    pass


# pytorch模板
def pytorch_test_template():
    image = torch.randn(4, 6, 128, 416)  # 输入尺寸 [N C H W]
    # function to be test
    pass


def dataflow_test():
    from DataFlow.sequence_folders import SequenceFolder
    import custom_transforms
    from torch.utils.data import DataLoader
    from DataFlow.validation_folders import ValidationSet
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([custom_transforms.RandomHorizontalFlip(),
                                                 custom_transforms.RandomScaleCrop(),
                                                 custom_transforms.ArrayToTensor(), normalize])
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
    datapath = 'G:/data/KITTI/KittiRaw_formatted'
    seed = 8964
    train_set = SequenceFolder(datapath, transform=train_transform, seed=seed, train=True,
                               sequence_length=3)

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4,
                              pin_memory=True)

    val_set = ValidationSet(datapath, transform=valid_transform)
    print("length of train loader is %d" % len(train_loader))
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    print("length of val loader is %d" % len(val_loader))

    dataiter = iter(train_loader)
    imgs, intrinsics = next(dataiter)
    print(len(imgs))
    print(intrinsics.shape)

    pass


# 深度指标

def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) #& (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred.squeeze()[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt) / torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred) ** 2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]


def errors_test():
    # x1 = torch.randn(4, 128, 416)
    # x2 = torch.randn(4, 1, 128, 416)

    depth0 = torch.from_numpy(np.load("H:/KittiRaw_prepare/2011_09_26_drive_0001_sync_02/0000000000.npy"))
    depth1 = torch.from_numpy(np.load("H:/KittiRaw_prepare/2011_09_26_drive_0001_sync_02/0000000001.npy"))
    depth2 = torch.from_numpy(np.load("H:/KittiRaw_prepare/2011_09_26_drive_0001_sync_02/0000000002.npy"))
    depth3 = torch.from_numpy(np.load("H:/KittiRaw_prepare/2011_09_26_drive_0001_sync_02/0000000003.npy"))

    x1 = torch.cat((depth0.unsqueeze(dim=0), depth1.unsqueeze(dim=0), depth2.unsqueeze(dim=0), depth3.unsqueeze(dim=0)), dim=0)
    # x1 = x1.unsqueeze(dim=1)
    x2 = torch.cat((depth3.unsqueeze(dim=0), depth2.unsqueeze(dim=0), depth1.unsqueeze(dim=0), depth0.unsqueeze(dim=0)), dim=0)
    x2 = x2.unsqueeze(dim=1)
    print(x1.shape)
    print(x2.shape)
    y = compute_errors(x1, x2)
    print(y)
    pass


def showdepthnpy():
    # depth0 = np.load("H:/KittiRaw_prepare/2011_09_26_drive_0001_sync_02/0000000000.npy")
    depth0 = np.load("F:/test.npy")

    depth1 = np.load("H:/KittiRaw_prepare/2011_09_26_drive_0001_sync_02/0000000001.npy")
    depth2 = np.load("H:/KittiRaw_prepare/2011_09_26_drive_0001_sync_02/0000000002.npy")
    depth3 = np.load("H:/KittiRaw_prepare/2011_09_26_drive_0001_sync_02/0000000003.npy")

    import cv2
    cv2.imshow("depth", depth0)
    print(depth0.max())
    print(depth0.min())
    cv2.waitKey(0)
    pass



import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.conv_1 = nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2,2))
        self.conv_2 = nn.Sequential(torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2,2))
        self.fc = nn.Sequential(nn.Linear(20*5*5, 200),
                                 nn.Linear(200, 100),
                                 nn.Linear(100, 10))

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self,x):
        out1 = self.conv_1(x)
        out2 = self.conv_2(out1)
        out2 = out2.view(-1, self.num_flat_features(out2))
        out3 = self.fc(out2)
        return out3

if __name__ == '__main__':
    model = Net()
    model = model.cuda()
    model.eval()

    image = torch.randn(1, 1, 28, 28)  # 输入尺寸 [N C H W]
    image = image.cuda()
    with torch.no_grad():
        out = model(image)

    print(out.shape)
    pass
