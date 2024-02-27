import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import os
import random
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Split CIFAR-10 dataset')

# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                     help='number of data loading workers (default: 4)')
# parser.add_argument('--epochs', default=300, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=128, type=int,
#                     metavar='N', help='mini-batch size (default: 128)')
def split_niid_nonreplace():
    label_strings = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    collection =[]
    path = './image_data/'
    save_path_train = './data_split_user/train_100_niid_0505'
    # save_path_train = './data_split_user/test'
    if os.path.exists(save_path_train) == False:
        os.makedirs(save_path_train)
    
    image_class_num = 5000
    # user_num = 720
    user_num = 100
    class_num = 10
    group_num = 10
    # user_num = 6
    # class_num = 2
    class_persentage = [0.5, 0.5]
    # user_image_num = [100, 200, 400]
    user_image_num = [500]
    image_num_preclass = 250
    
    user_train_data_index = np.zeros(user_num * class_num ).reshape(user_num, class_num)
    user_train_data = np.arange(user_num * user_image_num[len(user_image_num) - 1] * 3072, dtype='uint8').reshape(user_num, user_image_num[len(user_image_num) - 1], 32, 32, 3)
    user_train_labels = np.arange(user_num * (1 + user_image_num[len(user_image_num) - 1]), dtype = 'int32').reshape(user_num, (1 + user_image_num[len(user_image_num) - 1]))
    user_id = 0
    class_train_label = np.arange(image_class_num * 10, dtype='int32').reshape(10, image_class_num)
    class_train_label_num = np.arange(image_class_num * 10 + 1, dtype='int32')
    class_train_raw_data = np.arange(image_class_num * 3072 * 10, dtype='uint8').reshape(10, image_class_num, 3, 32, 32)
    class_train_data = np.arange(image_class_num * 3072 * 10, dtype='uint8').reshape(10, image_class_num, 32, 32, 3)

    #Load data from split dataset (different class)
    for i in range(10):
        fpath = os.path.join(path, 'train/' + label_strings[i] + '.bin')
        # fpath = os.path.join(path, 'test/' + label_strings[i] + '.bin')
        raw = np.fromfile(fpath, dtype='uint8')
        temp = np.reshape(raw, (image_class_num, 3073))
        
        class_train_label[i] = temp[:, 0]
        class_train_raw_data[i] = np.reshape(temp[:, 1:], (image_class_num, 3, 32, 32))
        temp = np.arange(1)
        for image_num in range(image_class_num):
            for color in range(3):
                for x in range(32):
                    for y in range(32):
                        class_train_data[i][image_num][x][y][color] = class_train_raw_data[i][image_num][color][x][y]
        temp_state = np.random.get_state()
        np.random.shuffle(class_train_data[i])
        np.random.set_state(temp_state)
        np.random.shuffle(class_train_label[i])
    # for user in range(user_num):
    #     for class in range(class_num):
            
    # file_name_data = save_path_train + '/test_data.bin'
    # file_name_labels = save_path_train + '/test_labels.bin'
    # class_train_data.flatten()
    # class_train_data.tofile(file_name_data)
    
    # class_train_label_num[0] = image_class_num * 10
    # class_train_label_num[1:] = class_train_label.flatten()
    # class_train_label_num.tofile(file_name_labels)
    # for i in range(1, 6):
    #     fpath = os.path.join(path, 'data_batch_' + str(i))
    #     raw = np.fromfile(fpath, dtype='uint8')
    #     collection.append(np.reshape(raw, (10000, 3073)))
    # records = np.concatenate(collection, axis=0)
    # labels = records[:, 0]
    # images = np.reshape(records[:, 1:], (50000, 3, 32, 32,))


    # Split dataset for different users
    #0-124 give to i, 125-249 give to j
    round_num = int(user_num/ (class_num*class_num))
    pre_class = int(len(class_persentage))
    user_train_data_index = np.zeros(class_num)
    class_start_index = np.zeros(class_num)
    # for k in range(round_num):
    #     for i in range(class_num):
    #         for j in range(class_num):
    #             user_index = i*class_num*round_num + j * round_num + k
    #             user_train_data[user_index][0: image_num_preclass] = class_train_data[i][int(user_train_data_index[i]): int(user_train_data_index[i]) + image_num_preclass]
    #             user_train_labels[user_index][0: image_num_preclass] = class_train_label[i][int(user_train_data_index[i]): int(user_train_data_index[i]) + image_num_preclass]
    #             user_train_data_index[i] +=image_num_preclass
            
    #             user_train_data[user_index][image_num_preclass: image_num_preclass * pre_class] = class_train_data[j][int(user_train_data_index[j]): int(user_train_data_index[j]) + image_num_preclass]
    #             user_train_labels[user_index][image_num_preclass: image_num_preclass * pre_class] = class_train_label[j][int(user_train_data_index[j]): int(user_train_data_index[j]) + image_num_preclass]
    #             user_train_data_index[j] +=image_num_preclass
                
    #             temp_state = np.random.get_state()
    #             np.random.shuffle(user_train_labels[user_index][1:])
    #             np.random.set_state(temp_state)
    #             np.random.shuffle(user_train_data[user_index])
    #             user_train_labels[user_index][0] = user_image_num[-1]
    #             raw_user_labels = user_train_labels[user_index].flatten()
    #             raw_user_data = user_train_data[user_index].flatten()
    #             file_name_data = save_path_train + '/user_' + str(user_index) + '.bin'
    #             file_name_labels = save_path_train + '/user_labels_' +str(user_index) + '.bin'    
    #             raw_user_labels.tofile(file_name_labels)
    #             raw_user_data.tofile(file_name_data)
    user_pre_group = user_num / group_num
    user_pre_group = int(user_pre_group)
    user_index = 0
    for class_index in range(class_num):
        for group_index in range(user_pre_group):
            start_index_1 = int(class_start_index[class_index])
            start_index_2 = int(class_start_index[(class_index + 1)%class_num])
            user_train_data[user_index][:image_num_preclass] = class_train_data[class_index][start_index_1: (start_index_1 + image_num_preclass)]
            user_train_labels[user_index][1:image_num_preclass] = class_index
            user_train_data[user_index][image_num_preclass:] = class_train_data[(class_index + 1)%class_num][start_index_2: (start_index_2 + image_num_preclass)]
            user_train_labels[user_index][image_num_preclass:] = (class_index + 1)%class_num
            class_start_index[class_index] = class_start_index[class_index] + image_num_preclass
            class_start_index[(class_index + 1)%class_num] = class_start_index[(class_index + 1)%class_num] + image_num_preclass
            temp_state = np.random.get_state()
            np.random.shuffle(user_train_labels[user_index][1:])
            np.random.set_state(temp_state)
            np.random.shuffle(user_train_data[user_index])
            user_train_labels[user_index][0] = user_image_num[-1]
            raw_user_labels = user_train_labels[user_index].flatten()
            raw_user_data = user_train_data[user_index].flatten()
            file_name_data = save_path_train + '/user_' + str(user_index) + '.bin'
            file_name_labels = save_path_train + '/user_labels_' +str(user_index) + '.bin'    
            raw_user_labels.tofile(file_name_labels)
            raw_user_data.tofile(file_name_data)
            user_index = user_index + 1


def split_iid():
    label_strings = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    collection =[]
    path = './image_data/'
    save_path_train = './data_split_user/train_100_iid_ns'
    # save_path_train = './data_split_user/test'
    if os.path.exists(save_path_train) == False:
        os.makedirs(save_path_train)
    
    image_class_num = 5000
    # user_num = 720
    user_num = 100
    class_num = 10
    # user_num = 6
    # class_num = 3
    user_image_num = [500]
    
    
    user_train_data_index = np.zeros(user_num * class_num ).reshape(user_num, class_num)
    user_train_data = np.arange(user_num * user_image_num[len(user_image_num) - 1] * 3072, dtype='uint8').reshape(user_num, user_image_num[len(user_image_num) - 1], 32, 32, 3)
    user_train_labels = np.arange(user_num * (1 + user_image_num[len(user_image_num) - 1]), dtype = 'int32').reshape(user_num, (1 + user_image_num[len(user_image_num) - 1]))
    user_id = 0
    class_train_label = np.arange(image_class_num * 10, dtype='int32').reshape(10, image_class_num)
    class_train_label_num = np.arange(image_class_num * 10 + 1, dtype='int32')
    class_train_raw_data = np.arange(image_class_num * 3072 * 10, dtype='uint8').reshape(10, image_class_num, 3, 32, 32)
    class_train_data = np.arange(image_class_num * 3072 * 10, dtype='uint8').reshape(10, image_class_num, 32, 32, 3)

    #Load data from split dataset (different class)
    for i in range(10):
        fpath = os.path.join(path, 'train/' + label_strings[i] + '.bin')
        # fpath = os.path.join(path, 'test/' + label_strings[i] + '.bin')
        raw = np.fromfile(fpath, dtype='uint8')
        temp = np.reshape(raw, (image_class_num, 3073))
        
        class_train_label[i] = temp[:, 0]
        class_train_raw_data[i] = np.reshape(temp[:, 1:], (image_class_num, 3, 32, 32))
        temp = np.arange(1)
        for image_num in range(image_class_num):
            for color in range(3):
                for x in range(32):
                    for y in range(32):
                        class_train_data[i][image_num][x][y][color] = class_train_raw_data[i][image_num][color][x][y]
        temp_state = np.random.get_state()
        np.random.shuffle(class_train_data[i])
        np.random.set_state(temp_state)
        np.random.shuffle(class_train_label[i])
    # for user in range(user_num):
    #     for class in range(class_num):
            
    # file_name_data = save_path_train + '/test_data.bin'
    # file_name_labels = save_path_train + '/test_labels.bin'
    # class_train_data.flatten()
    # class_train_data.tofile(file_name_data)
    
    # class_train_label_num[0] = image_class_num * 10
    # class_train_label_num[1:] = class_train_label.flatten()
    # class_train_label_num.tofile(file_name_labels)
    # for i in range(1, 6):
    #     fpath = os.path.join(path, 'data_batch_' + str(i))
    #     raw = np.fromfile(fpath, dtype='uint8')
    #     collection.append(np.reshape(raw, (10000, 3073)))
    # records = np.concatenate(collection, axis=0)
    # labels = records[:, 0]
    # images = np.reshape(records[:, 1:], (50000, 3, 32, 32,))


    # Split dataset for different users
    #0-124 give to i, 125-249 give to j
    image_pre_class_num = int(image_class_num/user_num)
        
    for user_id in range(user_num):
        for class_id in range (class_num):
            user_train_data[user_id][class_id * image_pre_class_num: (class_id + 1) * image_pre_class_num] = class_train_data[class_id][user_id * image_pre_class_num: (user_id + 1) * image_pre_class_num]
            user_train_labels[user_id][class_id * image_pre_class_num + 1: (class_id + 1) * image_pre_class_num + 1] = class_train_label[class_id][user_id * image_pre_class_num: (user_id + 1) * image_pre_class_num]
        # temp_state = np.random.get_state()
        # np.random.shuffle(user_train_labels[user_id][1:])
        # np.random.set_state(temp_state)
        # np.random.shuffle(user_train_data[user_id])
        user_train_labels[user_id][0] = user_image_num[-1]
        raw_user_labels = user_train_labels[user_id].flatten()
        raw_user_data = user_train_data[user_id].flatten()
        file_name_data = save_path_train + '/user_' + str(user_id) + '.bin'
        file_name_labels = save_path_train + '/user_labels_' +str(user_id) + '.bin'    
        raw_user_labels.tofile(file_name_labels)
        raw_user_data.tofile(file_name_data)

def split_test():
    label_strings = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    collection =[]
    path = './image_data/'
    # save_path_train = './data_split_user/train_100_niid_0505'
    save_path_train = './data_split_user/test'
    if os.path.exists(save_path_train) == False:
        os.makedirs(save_path_train)
    
    image_class_num = 1000
    # user_num = 720
    user_num = 100
    class_num = 10
    group_num = 10
    # user_num = 6
    # class_num = 2
    class_persentage = [0.5, 0.5]
    # user_image_num = [100, 200, 400]
    user_image_num = [500]
    image_num_preclass = 250
    
    user_train_data_index = np.zeros(user_num * class_num ).reshape(user_num, class_num)
    user_train_data = np.arange(user_num * user_image_num[len(user_image_num) - 1] * 3072, dtype='uint8').reshape(user_num, user_image_num[len(user_image_num) - 1], 32, 32, 3)
    user_train_labels = np.arange(user_num * (1 + user_image_num[len(user_image_num) - 1]), dtype = 'int32').reshape(user_num, (1 + user_image_num[len(user_image_num) - 1]))
    user_id = 0
    class_train_label = np.arange(image_class_num * 10, dtype='int32').reshape(10, image_class_num)
    class_train_label_num = np.arange(image_class_num * 10 + 1, dtype='int32')
    class_train_raw_data = np.arange(image_class_num * 3072 * 10, dtype='uint8').reshape(10, image_class_num, 3, 32, 32)
    class_train_data = np.arange(image_class_num * 3072 * 10, dtype='uint8').reshape(10, image_class_num, 32, 32, 3)

    #Load data from split dataset (different class)
    for i in range(10):
        fpath = os.path.join(path, 'test/' + label_strings[i] + '.bin')
        raw = np.fromfile(fpath, dtype='uint8')
        temp = np.reshape(raw, (image_class_num, 3073))
        
        class_train_label[i] = temp[:, 0]
        class_train_raw_data[i] = np.reshape(temp[:, 1:], (image_class_num, 3, 32, 32))
        temp = np.arange(1)
        for image_num in range(image_class_num):
            for color in range(3):
                for x in range(32):
                    for y in range(32):
                        class_train_data[i][image_num][x][y][color] = class_train_raw_data[i][image_num][color][x][y]
    test_data = class_train_data.flatten()
    test_labels = class_train_label.flatten()
    class_train_label_num[1:] = test_labels
    class_train_label_num[0] = image_class_num * class_num
    file_name_data = save_path_train + './test_data.bin'
    file_name_labels = save_path_train + './test_labels.bin'    
    test_data.tofile(file_name_data)
    class_train_label_num.tofile(file_name_labels)


def main():
    global args
    args = parser.parse_args()

    label_strings = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    collection =[]
    path = './image_data/'
    save_path_train = './data_split_user/train_100_niid'
    # save_path_train = './data_split_user/test'
    if os.path.exists(save_path_train) == False:
        os.makedirs(save_path_train)
    
    image_class_num = 5000
    # user_num = 720
    user_num = 100
    class_num = 10
    # user_num = 6
    # class_num = 3
    class_persentage = [0.5, 0.5]
    # user_image_num = [100, 200, 400]
    user_image_num = [500]
    image_num_preclass = 250
    user_train_data_index = np.zeros(class_num)
    
    user_train_data = np.arange(user_num * user_image_num[len(user_image_num) - 1] * 3072, dtype='uint8').reshape(user_num, user_image_num[len(user_image_num) - 1], 32, 32, 3)
    user_train_labels = np.arange(user_num * (1 + user_image_num[len(user_image_num) - 1]), dtype = 'int32').reshape(user_num, (1 + user_image_num[len(user_image_num) - 1]))
    user_id = 0
    class_train_label = np.arange(image_class_num * 10, dtype='int32').reshape(10, image_class_num)
    class_train_label_num = np.arange(image_class_num * 10 + 1, dtype='int32')
    class_train_raw_data = np.arange(image_class_num * 3072 * 10, dtype='uint8').reshape(10, image_class_num, 3, 32, 32)
    class_train_data = np.arange(image_class_num * 3072 * 10, dtype='uint8').reshape(10, image_class_num, 32, 32, 3)

    #Load data from split dataset (different class)
    for i in range(10):
        fpath = os.path.join(path, 'train/' + label_strings[i] + '.bin')
        # fpath = os.path.join(path, 'test/' + label_strings[i] + '.bin')
        raw = np.fromfile(fpath, dtype='uint8')
        temp = np.reshape(raw, (image_class_num, 3073))
        
        class_train_label[i] = temp[:, 0]
        class_train_raw_data[i] = np.reshape(temp[:, 1:], (image_class_num, 3, 32, 32))
        temp = np.arange(1)
        for image_num in range(image_class_num):
            for color in range(3):
                for x in range(32):
                    for y in range(32):
                        class_train_data[i][image_num][x][y][color] = class_train_raw_data[i][image_num][color][x][y]
        temp_state = np.random.get_state()
        np.random.shuffle(class_train_data[i])
        np.random.set_state(temp_state)
        np.random.shuffle(class_train_label[i])
    
            
    # file_name_data = save_path_train + '/test_data.bin'
    # file_name_labels = save_path_train + '/test_labels.bin'
    # class_train_data.flatten()
    # class_train_data.tofile(file_name_data)
    
    # class_train_label_num[0] = image_class_num * 10
    # class_train_label_num[1:] = class_train_label.flatten()
    # class_train_label_num.tofile(file_name_labels)
    # for i in range(1, 6):
    #     fpath = os.path.join(path, 'data_batch_' + str(i))
    #     raw = np.fromfile(fpath, dtype='uint8')
    #     collection.append(np.reshape(raw, (10000, 3073)))
    # records = np.concatenate(collection, axis=0)
    # labels = records[:, 0]
    # images = np.reshape(records[:, 1:], (50000, 3, 32, 32,))


    # Split dataset for different users
    
    user_index  = 0
    for i in range(class_num):
        for j in range(class_num):
            for k in range(image_num_preclass):
                user_train_data[user_index][k] = class_train_data[i][k + int(user_train_data_index[i])]
                user_train_labels[user_index][k + 1] = class_train_label[i][k + int(user_train_data_index[i])]
            user_train_data_index[i] += image_num_preclass
            for k in range(image_num_preclass):
                user_train_data[user_index][k + image_num_preclass] = class_train_data[j][k + int(user_train_data_index[j])]
                user_train_labels[user_index][k + image_num_preclass + 1] = class_train_label[j][k + int(user_train_data_index[j])]
            user_train_data_index[j] += image_num_preclass
            
            temp_state = np.random.get_state()
            np.random.shuffle(user_train_data[user_index])
            np.random.set_state(temp_state)
            np.random.shuffle(user_train_labels[user_index][1:-1])
            file_name_data = save_path_train + '/user_' + str(user_index) + '.bin'
            file_name_labels = save_path_train + '/user_labels_' +str(user_index) + '.bin'
            
            user_train_labels[user_index][0] = user_image_num[-1]
            raw_user_labels = user_train_labels[user_index].flatten()
            raw_user_data = user_train_data[user_index].flatten()
            
            raw_user_labels.tofile(file_name_labels)
            raw_user_data.tofile(file_name_data)
            user_index += 1
                # user_train_data[user_id] = np.random.choice(class_train_data[i], N_i)
                # user_train_labels[user_id] = class_train_label[i][0:image_num * class_persentage[0]]

                # user_train_data[user_id] = np.concatenate(user_train_data[user_id], np.random.choice(class_train_data[j], size=N_j))
                # user_train_labels[user_id] = np.concatenate(user_train_labels[user_id], class_train_label[i][0:image_num * class_persentage[1]])

                # user_train_data[user_id] = np.concatenate(user_train_data[user_id], np.random.choice(class_train_data[k], size=N_k))
                # user_train_labels[user_id] = np.concatenate(user_train_labels[user_id], class_train_label[i][0:image_num * class_persentage[2]])
                
    # record = user_train_data_index.flatten()
    # record.tofile(save_path_train + 'num.bin')
    

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, 4),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]), download=True),
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)
    # print(len(train_loader))
    # for i in range(10):
    #     train_subset = train_loader()

if __name__ == '__main__':
    # split_iid()
    # main()
    split_niid_nonreplace()
    # split_test()