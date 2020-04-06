import torch


if __name__ == '__main__':
    input = torch.empty(3, 5, 7, 9)
    batch_size = input.shape
    print(batch_size)