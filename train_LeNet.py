'''
Author       : ZHP
Date         : 2020-11-18 20:19:25
LastEditors  : ZHP
LastEditTime : 2020-11-23 21:16:12
FilePath     : /Earlier_Project/train_LeNet.py
Description  : LeNet训练 usage: python train_LeNet --batch_size=batch size --lr=lr 
Copyright 2020 ZHP
'''
# coding: utf-8
import os
import torch
import models
import json
import argparse
import copy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from PIL import Image
import time
from dataset import CMESet
from test_LeNet import test_one, test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='LeNet CME!')
    parser.add_argument('--originalDir', default='/disk/dataset/cme/pytorch/vgg/201101_modify/', help='dataset original image directory')
    parser.add_argument('--label_info', default='/disk/dataset/cme/pytorch/vgg/201101_modify_01_label.txt', help='dataset image label info file')
    parser.add_argument('--saveDir', default='/disk/dataset/Earlier_Project/model_result/', help='model save dir')
    
    parser.add_argument('--model_name', default='LeNet5', help='model name')
    parser.add_argument('--size', type=int, default=112, help='input resize size')

    parser.add_argument('--use_gpu', action='store_true', default=True, help=' use gpu ?')

    parser.add_argument('--nThreads', type=int, default=0, help='number of threads for data loading')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for train')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_final', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip', action='store_true', help='clip gradient?')
    parser.add_argument('--gradient', type=float, default=5, help='grad clip threshold')
    parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--show', type=tuple, default=(5, 3), help='number of epochs to show valid image')

    # test
    parser.add_argument('--test', action='store_true', default=False, help='test ?')
    parser.add_argument('--testDir', default='/media/disk3/zhp/dataset/cme_data/cme_matting/test/test_ori/', help='test orignal image datasets dir')
    parser.add_argument('--testresultDir', default='/media/disk3/zhp/dataset/cme_data/cme_matting/result/', help='test result dir')
    args = parser.parse_args()
    print('主要参数配置如下：\n')
    for key, value in args.__dict__.items():
        print(f'{key:^20} : {str(value):<}')
    return args


def set_learning_rate(optimizer, lr, end_lr, epoch, num):
    # update learning rate
    lr = lr * (0.5 ** (epoch // num))
    lr = end_lr if lr <= end_lr else lr
    for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def cal_time(total_time):
    h = total_time // 3600
    minute = (total_time % 3600) // 60
    sec = int(total_time % 60)
    print(f'本次总时长为： {h} hours {minute} minutes {sec} seconds...')


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


@torch.no_grad()
def valid(model, dataloader, criterion):
    """
    计算模型准确率
    """
    model.eval()

    val_loss = 0
    acc_count = 0
    total = 0
    # global device
    for ii, (image, label) in enumerate(dataloader):
        image = image.to(device)
        label = label.to(device)
        y_pre = model(image)  
        loss = criterion(y_pre, label)

        label_pre = torch.argmax(y_pre, dim=1)
        acc_count += (label == label_pre).sum().item()
        total += label.shape[0]
        val_loss += loss.item()
    model.train()
    val_loss = val_loss / (ii+1)
    acc = acc_count / float(total)
    return val_loss, acc


def train_Net(args):
    print("========================  Environment  ===============================================\n")
    if args.use_gpu:
        if torch.cuda.is_available():
            print("use GPU !")
        else:
            print("No GPU is avaiable !")
    else:
        print("use CPU !")
    result_info = copy.deepcopy(args.__dict__)
    # config model
    model = getattr(models, args.model_name)(num_classes=2, grayscale=True)
    model.to(device)
    lr = args.lr
    prefix = '{0}/batch_{1}_lr_{2}/'.format(args.model_name, args.batch_size, lr)
    save_model_dir = args.saveDir + prefix
    # 创建模型文件夹，并将训练中最好结果保存为json
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
        print(f'folder {save_model_dir} touch..')
    result_info['prefix'] = prefix
    result_info['save_model_dir'] = save_model_dir

    print("=====================  Loading datasets  =============================================\n")
    
    transform = transforms.Compose([
                transforms.Resize(args.size, 0),   # 插值方式采用Image.NEAREST，对二值图像不会产生0-1的灰度值
                transforms.ToTensor()  
            ])
    
    train_data = CMESet(root=args.originalDir, lists=args.label_info, transforms=transform) 
    # split valid set and train set
    n_train = int(len(train_data)*0.8)
    n_valid = len(train_data) - n_train
    train_set, valid_set = random_split(train_data, [n_train, n_valid])
    if args.test:
        test_data = CMESet(root=args.testDir, lists=args.label_info, transforms=transform, test=True)

    # dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, \
                                shuffle=True, num_workers=args.nThreads,\
                                    pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, \
                                shuffle=False, num_workers=args.nThreads,\
                                    pin_memory=True)
    
    # step2 ： criterion and optimiezr
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
    # update learning rate
    lambda1 = lambda epoch: lr * (0.5 ** (epoch // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    # train
    print('====================  start train  ==================================================\n')
    time_start = time.time()
    print('start at {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    
    # Configure tensorboard recording training data
    writer_name = 'runs/{}'.format(prefix)
    if not os.path.exists(writer_name):
        os.makedirs(writer_name)
    writer = SummaryWriter(log_dir=writer_name)
    print('\n可通过    tensorboard --logdir={}    来查看训练实时数据\n'.format(writer_name))
    text_info = ''
    for key, value in result_info.items():
        text_info += (f'{key:^20} : {str(value):<}\n')
    writer.add_text('train args', text_info, 1)
    train_start_info = copy.deepcopy(result_info)

    

    min_train_loss, loss_item, val_min_loss, best_acc = 1000, 0, 1000, 0
    train_lr = lr
    for epoch in range(1, args.nEpochs+1):
        train_iterator = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (image, label) in train_iterator:
            # data to GPU
            image = image.to(device)
            label = label.to(device)
            
            # forward
            y_pre = model(image)
            loss = criterion(y_pre, label)
            loss_item += loss.item()
      
            # backward
            optimizer.zero_grad()
            if args.clip:
                clip_gradient(optimizer, args.gradient)
            loss.backward()
            optimizer.step()

            train_lr = set_learning_rate(optimizer, lr, args.lr_final, epoch, 20) 
            train_iterator.set_description("train epoch {0}/{1},batch {2}".format(epoch, args.nEpochs, i+1))
    
        loss_item = loss_item / (i+1)
        writer.add_scalar(f'{args.model_name}/train_learning_rate', train_lr, epoch)
        writer.add_scalar(f'{args.model_name}/train loss', loss_item, epoch)
        
        val_loss, acc = valid(model, valid_loader, criterion)
        if acc > best_acc:
            best_acc = acc
            print('\nacc improved..，start save model ..')
            save_name = time.strftime(f'{save_model_dir}epoch{epoch}_' + '%m%d_%H_%M_%S.pth')
            torch.save(model.state_dict(), save_name)
            result_info['best_model_path'], result_info['best_ClassificationAccuracy'] = save_name, best_acc
        elif (loss_item < min_train_loss) and (epoch > (args.nEpochs - 10)):
            print('\nTraining is about to end, save the best model in the final stage ..')
            save_name = time.strftime(f'{save_model_dir}epoch{epoch}_' + '%m%d_%H_%M_%S.pth')
            torch.save(model.state_dict(), save_name)
            min_train_loss = loss_item
            result_info['best_train_loss'] = min_train_loss
        if loss_item < min_train_loss:
            min_train_loss = loss_item
            result_info['best_train_loss'] = min_train_loss
     
        print(f'\ntrain loss : {loss_item :^6.4f}     valid loss : {val_loss:.3f}   valid Classification Accuracy : {acc*100:.3f}%\n')
        
        writer.add_scalar(f'{args.model_name}/valid_loss', val_loss, epoch)
        writer.add_scalar(f'{args.model_name}/valid_CA', acc, epoch)
        loss_item = 0
    # 训练结束关闭监控
    text_info = ''
    for key, value in (result_info.items() - train_start_info.items()):
        text_info += (f'{key:^20} : {str(value):<}\n')
    writer.add_text('train args', text_info, 2)
    writer.close()

    # 写入训练详细信息
    result_path = save_model_dir + 'result.json'
    with open(result_path, 'w') as f:
        json.dump(result_info, f, indent=4)
    print(f'训练即将结束，本次训练详细信息保存在 :{result_path}\nbest Classification Accuracy : {best_acc*100:.3f}%\n')
    print('==============' + '   end train   ' + "======================\n")
    time_end = time.time()
    print('end at {}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    cal_time(time_end - time_start)
    

    if args.test :
        test_result_dir = args.testresultDir + prefix
        test(result_path, args.testDir, save_dir=test_result_dir, result_path='/disk/dataset/Earlier_Project/update_label.json')
        print('test done..')


if __name__ == '__main__':

    print("========================  loading args  ==============================================\n")
    global args
    args = get_args() 
    train_Net(args)  
