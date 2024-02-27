import numpy as np
import subprocess as sp
import multiprocessing as mp
import socket
import threading as td
import argparse
import os

from numpy.core.arrayprint import set_string_function
import vgg
import time
import copy
from collections import Counter

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
import SL_local_train as SL_local



torch.cuda.empty_cache()

model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and callable(vgg.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg19)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--local_epochs', default=5, type=int, metavar='N',
                    help='number of local client to train models')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--b', dest = 'batch_size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--id', dest = 'user_id', default=0, type=int,
                    metavar='UR', help='user_id of each client')
parser.add_argument('--user', dest = 'selected_user_num', default=10, type=int,
                    metavar='SU', help='The number of selected users')
parser.add_argument('--avauser', dest = 'user_num', default=100, type=int,
                    metavar='SU', help='The number of selected users')
parser.add_argument('--save_dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--data_path', dest='data_path',
                    help='The directory used to save the trained models',
                    default='./data_spilt_user/train_10_iid', type=str)
parser.add_argument('--process', dest='process_limit',
                    help='The maximum number of process to run in the GPU',
                    default=1, type=int)
parser.add_argument('--uplr', dest='update_lr', type=float, 
                    help='The directory used to save the trained models',
                    default=0.1)
parser.add_argument('--lrdecay', dest='lrdecay',
                    help='The number of round-decay',
                    default=50, type=int)
parser.add_argument('--group', dest='group', type=int, 
                    help='The size of group user',
                    default=1)
parser.add_argument('--iid', dest='iid', type=int, 
                    help='IID is 1 and non-IID is -1',
                    default=1)

# def local_training(user_id, model_save_dir, batch_size, learning_rate, model_load_dir=None):
#     if model_load_dir == None:
#         local_train = sp.run("python SL_local_train.py --arch=" + args.arch + " --epochs=" + str(args.local_epochs) + " --lr=" + str(learning_rate) + " --id=" + str(user_id) + ' --save-dir=' + model_save_dir + ' --b=' + str(batch_size) + ' --data_path=' + args.data_path, shell=True, stdout=sp.PIPE)
#         print('local training return code: ', local_train.returncode)
#     else:
#         local_train = sp.run("python SL_local_train.py --arch=" + args.arch + " --epochs=" + str(args.local_epochs) + " --lr=" + str(learning_rate) + " --id=" + str(user_id) + ' --resume=' + model_load_dir + ' --save-dir=' + model_save_dir + ' --b=' + str(batch_size) + ' --data_path=' + args.data_path, shell=True, stdout=sp.PIPE)
#         local_train.wait()


global args
args = parser.parse_args()
model_name = args.arch
user_num = args.user_num
class_num = 10
label_num = 5000
user_ava_num = user_num
learning_decay = 1.1
        

def user_selection(ava_user_list, data_path, class_num, group_num, start_index):
    label_distribution = np.zeros((len(ava_user_list), class_num))
    # for i in range(len(ava_user_list)):
    #     label_file = data_path + '/user_labels_' + str(i) + '.bin'
    #     label = np.fromfile(label_file, dtype='int32')
    #     for j in range(class_num):
    #         label_distribution[i, j] = Counter(label)[j]/(len(label) - 1)
    # print(label_distribution[33])
    group_size = int(len(ava_user_list) / group_num)
    selected_user_list = np.zeros([group_size], dtype = int)
    for i in range(group_size):
        selected_user_list[i] = start_index + i
    return selected_user_list

    


def trainProcess_selected(user_list, save_dir, model_name, user_selected_num, epoch, learning_rate):
    
    # user_list = all_user_list
    # user_list = all_user_list
    np.random.shuffle(user_list)
    print(user_list)
    
    
    if len(user_list) == 0 or args.epochs <= 0:
        exit(-1)
    # Split training of first epoch
    global_model = vgg.__dict__[model_name]()
    global_model.features = torch.nn.DataParallel(global_model.features)
    global_model.classifier = torch.nn.DataParallel(global_model.classifier)

    updated_global_model = vgg.__dict__[model_name]()
    updated_global_model.features = torch.nn.DataParallel(updated_global_model.features)
    updated_global_model.classifier = torch.nn.DataParallel(updated_global_model.classifier)
   
    local_model = vgg.__dict__[model_name]()
    local_model.features = torch.nn.DataParallel(local_model.features)
    local_model.classifier = torch.nn.DataParallel(local_model.classifier)
    
    
    if args.cpu:
        global_model.cpu()
        updated_global_model.cpu()
        local_model.cpu()

    else:
        global_model.cuda()
        updated_global_model.cuda()
        local_model.cuda()
    # client_thread_list = []
    # total_sample_num = args.selected_user_num * label_num
    total_sample_num =0
    if args.start_epoch != 0:
        epoch = epoch + args.start_epoch
        print('Training epochs = ', epoch)
        for user_id in user_list:
            if user_id == user_list[0]:
                SL_local.local_train(model_name, args.save_dir, os.path.join(args.save_dir,'checkpoint_avg_{}.tar'.format(epoch)), args.data_path, user_id, learning_rate, args.local_epochs, epoch, args.batch_size, os.path.join(args.save_dir,'checkpoint_avg_{}.tar'.format(epoch - 1)))
            else:
                SL_local.local_train(model_name, args.save_dir, os.path.join(args.save_dir,'checkpoint_avg_{}.tar'.format(epoch)), args.data_path, user_id, learning_rate, args.local_epochs, epoch, args.batch_size, os.path.join(args.save_dir,'checkpoint_avg_{}.tar'.format(epoch)))
    else: 
        if epoch == 0:
            print('Training epochs = ', epoch)
            user_id = user_list[0]
            
            for user_id in user_list:
                if user_id == user_list[0]:
                    SL_local.local_train(model_name, args.save_dir, os.path.join(args.save_dir,'checkpoint_avg_{}.tar'.format(epoch)), args.data_path, user_id, learning_rate, args.local_epochs, epoch, args.batch_size)
                else:
                    SL_local.local_train(model_name, args.save_dir, os.path.join(args.save_dir,'checkpoint_avg_{}.tar'.format(epoch)), args.data_path, user_id, learning_rate, args.local_epochs, epoch, args.batch_size, os.path.join(args.save_dir,'checkpoint_avg_{}.tar'.format(epoch)))
            
            torch.cuda.empty_cache()       
        else:
            print('Training epochs = ', epoch)
            for user_id in user_list:
                if user_id == user_list[0]:
                    SL_local.local_train(model_name, args.save_dir, os.path.join(args.save_dir,'checkpoint_avg_{}.tar'.format(epoch)), args.data_path, user_id, learning_rate, args.local_epochs, epoch, args.batch_size, os.path.join(args.save_dir,'checkpoint_avg_{}.tar'.format(epoch - 1)))
                else:
                    SL_local.local_train(model_name, args.save_dir, os.path.join(args.save_dir,'checkpoint_avg_{}.tar'.format(epoch)), args.data_path, user_id, learning_rate, args.local_epochs, epoch, args.batch_size, os.path.join(args.save_dir,'checkpoint_avg_{}.tar'.format(epoch)))
            torch.cuda.empty_cache()
    return user_list


selected_user_num = args.selected_user_num
global_distribution = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
file_path=args.data_path

def main():
    torch.cuda.empty_cache()
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)
    

    total_user_list = np.arange(user_num)
    # total_user_list = np.arange(5)
    # selected_user_list = user_selection(user_list=total_user_list, selected_user_num=selected_user_num, label_num=label_num, file_path='./data_split_user/train/')
    # print(selected_user_list)
    # trainProcess(selected_user_list, args.port, save_dir=args.save_dir, model_name=model_name)
    # total_user_list = np.arange(100)
    pre_user_list = []
    learning_rate = args.lr
    total_time_list = np.arange(args.epochs)
    group_num = 10
    group_size = 1
    if args.start_epoch != 0:
        iterative_num = (args.start_epoch + 1) / args.lrdecay
        iterative_num = int(iterative_num)
        for i in range(iterative_num):
            learning_rate = learning_rate ** learning_decay
    
    for i in range(args.epochs):
        if (i + 1) % args.lrdecay == 0:
            learning_rate = learning_rate ** learning_decay
        start_time = time.time()
        data_path = ''
        start_index = (i%group_num)*group_size
        selected_user_list = user_selection(total_user_list, data_path, class_num, group_num, start_index)
        if args.group == 1:
            # selected_user_list, ava_user_list = trainProcess(total_user_list, pre_user_list,args.port, save_dir=args.save_dir, model_name=model_name,user_selected_num=selected_user_num,epoch=i,learning_rate=learning_rate)
            # ava_user_list = np.random.choice(total_user_list, user_ava_num, replace=False)
            
            user_list = np.random.choice(total_user_list, args.selected_user_num, replace=False)
            mid_time = time.time()
            # print('Selected training time: ', mid_time - start_time)
            # time.sleep(2)
            if args.iid == 1:
                trainProcess_selected(user_list, save_dir=args.save_dir, model_name=model_name,user_selected_num=selected_user_num,epoch=i,learning_rate=learning_rate)
            else:
                trainProcess_selected(selected_user_list, save_dir=args.save_dir, model_name=model_name,user_selected_num=selected_user_num,epoch=i,learning_rate=learning_rate)
            
            end_time = time.time()
            print('Random selected time: ', end_time - mid_time)
            time.sleep(2)
            
        # pre_user_list = np.concatenate((pre_user_list, selected_user_list), axis=0)
        # pre_user_list = pre_user_list.flatten()
        # pre_user_list = np.unique(pre_user_list)
        
        total_time_list[i] = end_time - start_time
    total_time_list.tofile('total_time.txt')
        
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename, _use_new_zipfile_serialization= True)

def test():
    print('start')
    torch.cuda.empty_cache()

    model = vgg.__dict__[model_name]()
    model.features = torch.nn.DataParallel(model.features)
    model.classifier = torch.nn.DataParallel(model.classifier)
   
    model_avg = vgg.__dict__[model_name]()
    model_avg.features = torch.nn.DataParallel(model_avg.features)
    model_avg.classifier = torch.nn.DataParallel(model_avg.classifier)

    # for key in model_avg.state_dict().keys():
    #     print(model_avg.state_dict()[key])
    #     break

    filename=os.path.join(args.save_dir,'checkpoint_66.tar')
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    for key in model.state_dict().keys():
        print(model.state_dict()[key])
        break

    # filename=os.path.join(args.save_dir,'checkpoint_avg_0.tar')
    # checkpoint = torch.load(filename)
    # model_avg.load_state_dict(checkpoint['state_dict'])
    # model_avg.eval()
    # for key in model_avg.state_dict().keys():
    #     print(model_avg.state_dict()[key])
    #     break

    # filename=os.path.join(args.save_dir,'checkpoint_avg_1.tar')
    # checkpoint = torch.load(filename)
    # model_avg.load_state_dict(checkpoint['state_dict'])
    # model_avg.eval()
    # for key in model_avg.state_dict().keys():
    #     print(model_avg.state_dict()[key])
    #     break

    
    for key in model_avg.state_dict().keys():
        new = model.state_dict()[key] + model.state_dict()[key] 
        model.state_dict()[key].copy_(new)
        print(model.state_dict()[key])
        # newnew = model_avg.state_dict()[key] * 2
        # model_avg.state_dict()[key]=newnew.clone()
        # print(model_avg.state_dict()[key])
        break 
    

if __name__ == '__main__':
    main()