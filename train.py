import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from tvn.solver import Solver
from tvn.model import TVN, no_verbose
from tvn.config import CFG1
from tvn.data import SomethingSomethingV2
from torch.multiprocessing import cpu_count


if __name__ == '__main__':
    no_verbose()
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', default='./data', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--logs_path', default='./logs', type=str)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--resume', default=0, type=int)
    args = parser.parse_args()

    train_dataset = SomethingSomethingV2(root=args.data_root, mode='train')
    valid_dataset = SomethingSomethingV2(
        root=args.data_root, mode='validation')
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=cpu_count())

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=cpu_count())

    tvm = TVN(CFG1, num_classes=len(train_dataset.classes))
    device = torch.device('cuda')
    tvm = tvm.to(device)
    tvm = torch.nn.DataParallel(tvm)
    cudnn.benchmark = True

    start_epoch = -1
    optimizer = torch.optim.Adam(tvm.parameters())

    if args.resume == -1:
        path_checkpoint = "logs/ckpt/model-latest.pt"
        checkpoint = torch.load(path_checkpoint)

        tvm.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    solver = Solver(args.logs_path,
                    model=tvm,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    start_epoch=start_epoch+1,
                    optimizer=optimizer)
    # solver.test()
    for epoch in range(start_epoch+1, args.num_epochs):
        print("Epoch %d :" % epoch)
        solver.train()
        solver.test()
        solver.save()
