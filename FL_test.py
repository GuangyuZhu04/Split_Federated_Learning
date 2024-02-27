import numpy as np
import subprocess as sp
import threading as td
import argparse
import os
import vgg
import time
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and callable(vgg.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg11',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg11)')

parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--save_dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save_dir_iid', dest='save_dir_iid',
                    help='The directory used to save the trained models',
                    default='save_temp_iid', type=str)
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--set', dest='test_set', default=None, type=int, 
                    help='Set testing dataset in split format')
parser.add_argument('--print_freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('--id', dest = 'user_id', default=0, type=int,
                    metavar='UR', help='user_id of each client')
parser.add_argument('--save_id', dest='save_id', default=0, type=int,
                    metavar='N', help='The model ID of the trained model')
parser.add_argument('--model_num', dest='model_num', default=1, type=int,
                    metavar='N', help='The number of the trained model')

global args, best_prediction
args = parser.parse_args()


class Dataset_SF(Dataset):
    def __init__(self, label_file, img_dir, transform=None, target_transform=None):
        self.image_labels = np.fromfile(label_file, dtype='int32')
        self.img_dir = img_dir
        raw_image = np.fromfile(self.img_dir, dtype='uint8')
        # self.image = np.reshape(raw_image, (-1, 3, 32, 32))
        self.image = np.reshape(raw_image, (-1,32, 32, 3))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.image_labels[0]
    
    def __getitem__(self, idx):
        image = self.image[idx]
        label = np.int64(self.image_labels[idx + 1])
        im = Image.fromarray(np.uint8(image)).convert('RGB')
        if self.transform:
            image = self.transform(im)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    

    end = time.time()
    # correct = torch.zeros(1).squeeze().cuda()
    # total = torch.zeros(1).squeeze().cuda()
    loss_all = 0
    for i, (input, target) in enumerate(val_loader):
        if args.cpu == False:
            # input = input.cuda(async=True)
            # target = target.cuda(async=True)
            input = input.cuda()
            target = target.cuda()
        if args.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
        
        output = output.float()
        loss = loss.float()
        
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        loss_all += losses.avg
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        torch.cuda.empty_cache()
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
    
    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return float(top1.avg), loss_all / len(val_loader)



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    
    return res

save_id = args.save_id

def testing(model_index, save_path):
    criterion = nn.CrossEntropyLoss()
    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.test_set == None:
        #using training dataset
        # val_loader = torch.utils.data.DataLoader(
        # datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomCrop(32, 4),
        #     transforms.ToTensor(),
        #     normalize,
        # ]), download=True),
        # batch_size=args.batch_size, shuffle=True,
        # num_workers=args.workers, pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else: 
        if args.test_set == -1:
            dataset_user = Dataset_SF(label_file='./data_split_user/test/test_labels.bin',img_dir='./data_split_user/test/test_data.bin', transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]))
            val_loader = torch.utils.data.DataLoader(
                dataset_user,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        else:
            dataset_user = Dataset_SF(label_file='./data_split_user/test/user_labels_' + str(model_index) + '.bin',img_dir='./data_split_user/test/user_' + str(model_index) + '.bin', transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]))
            val_loader = torch.utils.data.DataLoader(
                dataset_user,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)


    model = vgg.__dict__[args.arch]()
    model.features = torch.nn.DataParallel(model.features)
    model.classifier = torch.nn.DataParallel(model.classifier)
    if args.cpu:
        model.cpu()
    else:
        model.cuda()
    filename=os.path.join(save_path, 'checkpoint_avg_{}.tar'.format(model_index))
    # filename=os.path.join(save_path, 'checkpoint_{}.tar'.format(model_index))
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['epoch'])

    precision, loss = validate(val_loader, model, criterion)
    print('Model precision: ', precision)
    print('Model average loss: {loss:.3f}', loss)
    return precision, loss


def main():
    pres_list = np.zeros(args.model_num)
    loss_list = np.zeros(args.model_num)
    pres_list_iid = np.zeros(args.model_num)
    loss_list_iid = np.zeros(args.model_num)
    if args.model_num == 1:
        testing(args.save_id, args.save_dir)
    else:
        for model_index in range(args.model_num):
            save_path = args.save_dir
            pres_list[model_index], loss_list[model_index] = testing(model_index, save_path)
            # save_path = args.save_dir_iid
            # pres_list_iid[model_index], loss_list_iid[model_index] = testing(model_index, save_path)
        pres_list = pres_list.flatten()
        loss_list = loss_list.flatten()
        pres_list.tofile('model_pres'+ str(args.model_num) + '_0.bin')
        loss_list.tofile('model_loss'+ str(args.model_num) + '_0.bin')
    




if __name__ == '__main__':
    main() 
