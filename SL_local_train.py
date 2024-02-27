import argparse
import os
import shutil
import time
import struct
import pickle
import numpy as np
from cifar10_noniid import cifar_extr_noniid, get_dataset_cifar10_extr_noniid

from PIL import Image
from matplotlib import cm
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
import vgg



torch.cuda.empty_cache()

model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and callable(vgg.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg11',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg19)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--b', dest = 'batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--id', dest = 'user_id', default=0, type=int,
                    metavar='UR', help='user_id of each client')
parser.add_argument('--port', dest = 'port', default=9998, type=int,
                    metavar='PT', help='Socket prot to transmit data')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--data_path', dest='data_path',
                    help='The directory used to save the trained models',
                    default='./data_split_user/train_100_iid', type=str)
parser.add_argument('--group', dest='group', type=int, 
                    help='The size of group user',
                    default=1)
parser.add_argument('--group_list', dest='group_list', type=list, 
                    help='The size of group user',
                    default=1)




class Dataset_SF(Dataset):
    def __init__(self, label_file, img_dir, transform=None, target_transform=None):
        self.image_labels = np.fromfile(label_file, dtype='int32')
        self.img_dir = img_dir
        raw_image = np.fromfile(self.img_dir, dtype='uint8')
        self.image = np.reshape(raw_image, (-1, 32, 32, 3))
        # self.image = np.reshape(raw_image, (-1, 3, 32, 32))
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

class Dataset_SF_group(Dataset):
    def __init__(self, label_file, img_dir, transform=None, target_transform=None):
        self.image_labels = np.fromfile(label_file[0], dtype='int32')
        self.image_num = self.image_labels[0]
        for label_dir in label_file[1:]:
            self.image_labels = np.concatenate(self.image_labels, np.fromfile(label_dir, dtype='int32')[1:])
            self.image_num += np.fromfile(label_dir, dtype='int32')[0]
        raw_image = np.fromfile(img_dir, dtype='uint8')[0]
        for img_dir in img_dir[1:]:
            raw_image = np.concatenate((raw_image, np.fromfile(img_dir, dtype='uint8')[1:]), axis=0)
        self.image = np.reshaperaw_image(raw_image, (-1, 32, 32, 3))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.image_num
    
    def __getitem__(self, idx):
        image = self.image[idx]
        label = np.int64(self.image_labels[idx + 1])
        im = Image.fromarray(np.uint8(image)).convert('RGB')
        if self.transform:
            image = self.transform(im)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# global args, best_prediction
# args = parser.parse_args()

best_prediction = 0
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


def local_train(arch, save_dir, model_save_name, data_path, user_id, lr, local_epochs, epochs, batch_size,  resume=None):
    group = 1
    workers = 4
    evaluate=False
    momentum = 0.9
    weight_decay = 5e-4
    # Check the save_dir exists or not
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = vgg.__dict__[arch]()
    model.features = torch.nn.DataParallel(model.features)
    model.classifier = torch.nn.DataParallel(model.classifier)
    
    
    model.cuda()
    
    if resume:
        if os.path.isfile(resume):
            # print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            # args.start_epoch = checkpoint['epoch']
            sample_num = checkpoint['sample_number']
            model.load_state_dict(checkpoint['state_dict'])
            # print("=> loaded checkpoint (epoch {})"
            #       .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    # The setting of training dataset and normalization 
    cudnn.benchmark = True

    if group == 1:
        dataset_user = Dataset_SF(label_file= data_path + '/user_labels_' + str(user_id) + '.bin',img_dir=data_path + '/user_' + str(user_id) + '.bin', transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        if group < 0:
            print('group number error')
            exit(-2)
        label_file = data_path + '/user_group_labels_' + str(user_id) + '.bin'
        img_dir = data_path + '/user_group_' + str(user_id) + '.bin'
        dataset_user = Dataset_SF(label_file=label_file, img_dir=img_dir, transform=transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]))
    train_loader = torch.utils.data.DataLoader(dataset_user,batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, 4),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]), download=True),
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    num_users_cifar = 10
    nclass_cifar = 10
    nsamples_cifar = 200
    rate_unbalance_cifar = 1.0

    # trainset, testset, user_groups_train_cifar, user_groups_test_cifar = get_dataset_cifar10_extr_noniid(num_users_cifar, nclass_cifar, nsamples_cifar, rate_unbalance_cifar)
    # train_loader = torch.utils.data.DataLoader(
    #     user_groups_train_cifar,
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)


    # Training parameter setting (loss function and optimizer)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    
    

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    if evaluate:
        print('validate')
        validate(val_loader, model, criterion)
        return
    

    for epoch in range(local_epochs):
        # ad_lr = adjust_learning_rate(lr, optimizer, epoch)
        
        prediction_rate = train(train_loader, model, criterion, optimizer, epoch)
        
        
        # prediction_rate = validate(val_loader, model, criterion)
        
        # remember best prec@1 and save checkpoint
        # is_best = prediction_rate > best_prediction
        # best_prediction = max(prediction_rate, best_prediction)
    
    save_checkpoint({
            'epoch': epochs,
            'state_dict': model.state_dict(),
            'sample_number': len(dataset_user),
            }, filename=model_save_name)
    
    torch.cuda.empty_cache()


# Client-sdie training function
def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    # Send the size of the dataset
    # datasets_size = 50000
    
    total_time = 0
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        print_freq = 5
        optimizer.zero_grad()
        
        input = input.cuda()
        target = target.cuda()

        # optimizer.zero_grad()
        # compute output
        output = model(input)
        
        # compute gradient and do SGD step
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        #measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        
        # measure elapsed time
        
        # if i % print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #               epoch, i, len(train_loader), batch_time=batch_time,
        #               data_time=data_time, loss=losses, top1=top1))
    # torch.cuda.empty_cache()
    # for i, (input, target) in enumerate(train_loader):
    #     # measure data loading time
    #     data_time.update(time.time() - end)

    #     if args.cpu:
    #         input = input.cpu()
    #         target = target.cpu()
    #     else:
    #         input = input.cuda()
    #         target = target.cuda()
            
    #     if args.half:
    #         input = input.half()

    #     # compute output
    #     output = model(input)
        
  
    #     # compute gradient and do SGD step
        
    #     optimizer.step()

    #     output = output.float()
    #     loss = client_grad.float()
    #     # measure accuracy and record loss
    #     # prec1 = accuracy(output.data, target)[0]
    #     # losses.update(loss.item(), input.size(0))
    #     # top1.update(prec1.item(), input.size(0))
        
        
    #     # if i % args.print_freq == 0:
    #     #     print('Epoch: [{0}][{1}/{2}]\t'
    #     #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #     #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
    #     #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #     #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
    #     #               epoch, i, len(train_loader), batch_time=batch_time,
    #     #               data_time=data_time, loss=losses, top1=top1))
    #     torch.cuda.empty_cache()
    # print('average time = ', total_time / len(train_loader))
    return 0


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
        torch.cuda.empty_cache()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename, _use_new_zipfile_serialization= True)

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


def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    ad_lr = lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = ad_lr
    return ad_lr


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
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    local_train()