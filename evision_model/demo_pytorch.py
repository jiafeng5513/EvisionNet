import torch


if __name__ == '__main__':
    input = torch.empty(3, 5, 7, 9)
    N,C,H,W = input.shape
    shape_list = [N,C,H,W]

    print(shape_list[-1:])