# -*- coding: utf-8 -*-
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import datetime
from torchvision import models
import os
import sys
from models import Models
import argparse

sys.path.append(".")
from data import genData

def validation(net, valid_data, loss_fc, is_cuda):

    loss_list = []
    acc_list = []
    with torch.no_grad():   # 不进行反向梯度传播
        st_time = time.time()
        for idx, batch in enumerate(valid_data):
            # get input
            inputs, label = batch
            
            inputs = Variable(inputs)
            label = Variable(label.long())
            
            if(is_cuda):
                inputs = inputs.cuda()
                label = label.cuda()
            
            # forward
            output = net(inputs)
   
            _, preds = torch.max(output.data, 1)
            loss_val = loss_fc(output, label)
            acc = torch.sum(preds == label).float() / len(batch[1])
            
            loss_list.append(float("{:.5f}".format(loss_val)))
            acc_list.append(float("{:.5f}".format(acc)))
            
        end_time = time.time()
        ave_loss = sum(loss_list) / (idx + 1)
        ave_acc = sum(acc_list) / (idx + 1)
        
        print("Validation... cost time: {}   ave loss: {:.5f} ave acc: {}".format(end_time -st_time,
                                                                             ave_loss, ave_acc))
        return ave_acc

def train_model(batch_size, lr, dataSet_train, dataSet_valid, ckpt_path, backbones, 
                st_epoch=0, 
                end_epoch=300,
                is_fintuning=False,
                is_cuda=True,
                frequent=10,
                optim="SGD"):
    '''
      Args:
         - backbones:主干网络, eg: "mobileNetV1" --> str, 注：模型保存格式以backbone命名 --> backbone_epoch.pt    
         - st_epoch:选择开始的轮数, eg: 0 --> int
         - is_fintuning:是否fintuning预训练的模型
         - frequent: the frequent of print log information
    '''
    #print("start train...")
    print("batch_size={}, lr={}, dataSet_train={}, dataSet_valid={}, ckpt_path={}, backbones={}, st_epoch={}, end_epoch=300, is_fintuning={}, is_cuda={}, frequent={}, optim={}".format(batch_size, 
                lr, 
                len(dataSet_train),
                len(dataSet_valid), 
                ckpt_path, 
                backbones, 
                st_epoch, 
                end_epoch,
                is_fintuning,
                is_cuda,
                frequent,
                optim))
    
    if(not os.path.exists(ckpt_path)):
        os.mkdir(ckpt_path)
    
#    net = models.resnet18(pretrained=True)
#    num_ftrs = net.fc.in_features
#    net.fc = nn.Linear(num_ftrs, 10)
    
    net = Models(backbones, embeding_size=128)

    if torch.cuda.is_available():
        net = net.cuda()
#    print(is_fintuning)
    if(is_fintuning):
        # 加载预训练参数
        pretrained_dict = torch.load(os.path.join(ckpt_path, backbones+"_%d.pt"%st_epoch))
        model_dict = net.state_dict()       # 模型的参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        st_epoch = st_epoch + 1
    
    if(is_cuda):
        net.cuda()
        
    loss_fc = nn.CrossEntropyLoss()         # loss fc
    
    assert optim in ["Adam", "SGD"]
    if(optim == "Adam"):
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001) 
    elif(optim == "SGD"):
        optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=0.0001)
        
    # generation data
    train_data = torch.utils.data.DataLoader(dataset=dataSet_train,
                                             batch_size=batch_size,
                                             num_workers=4,
                                             shuffle=True)
    
    #valid_data = dataSet_train.dataLoader()
    valid_data = torch.utils.data.DataLoader(dataset=dataSet_valid,
                                             batch_size=batch_size,
                                             num_workers=4,
                                             shuffle=False)
    
    best_vaild_acc = float("-inf")        # 保留验证集最好的准确率
    for epoch in range(st_epoch, end_epoch):
        print()
        
        net.train()
        acc_list = []
        loss_list = []
        
        st_time = time.time()
        
        for idx, batch in enumerate(train_data):
            
            # get input
            inputs, label = batch
            
            inputs = Variable(inputs)
            label = Variable(label.long())
            
            if(is_cuda):
                inputs = inputs.cuda()
                label = label.cuda()
            
            # forward
            output = net(inputs)
            #print("output:", output.data) # output: [batch_size, class_num], type:tensor
            #print("output max", torch.max(output.data, 1)) # ([max_value], [max_index])
            
            _, preds = torch.max(output.data, 1)
            loss_val = loss_fc(output, label)
            acc = torch.sum(preds == label).float() / len(batch[1])
            
            loss_list.append(float("{:.5f}".format(loss_val)))
            acc_list.append(float("{:.5f}".format(acc)))
            
            if(idx % frequent == 0):
                print("Epoch:{} time: {}  speed: {}/{} loss: {:.5f} acc: {:.5f} lr:{} optimizer: {}".format(epoch,
                                                              datetime.datetime.now(),
                                                              idx*batch_size, len(dataSet_train),                   
                                                              loss_val, acc, lr, optim))
            
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        
        end_tiem = time.time()
        ave_loss = sum(loss_list) / (idx + 1)
        ave_acc = sum(acc_list) / (idx + 1)
        
        print("Epoch {} cost time: {:.5f}s, ave loss: {:.5f} ave acc: {:.5f}, lr: {}, optimizer: {}".format(epoch, 
                                                              end_tiem-st_time,
                                                              ave_loss, ave_acc, lr, optim))
        
        # Validation
        print("*"*20, " Start Validation ", "*"*20)
        net.eval()
        valid_acc = validation(net, valid_data, loss_fc, is_cuda)
        
        # save model
        if(valid_acc > best_vaild_acc):
            best_vaild_acc = valid_acc
            save_path = os.path.join(ckpt_path, backbones+"_%d.pt"%epoch)
            torch.save(net.state_dict(), save_path)

def parse_args():
    
    parse = argparse.ArgumentParser(description="Train net config",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parse.add_argument("--is_train", dest="is_train", help="whether train or test",
                       default=True, type=str)
    parse.add_argument("--batch_size", dest="batch_size", help="train model choice batch size", 
                       default=128, type=int)
    parse.add_argument("--lr", dest="lr", help="learning rate",
                       default=0.001, type=float)
    parse.add_argument("--ckpt_path", dest="ckpt_path",help="model store directory",
                       default="./ckpt/mobilenetV2", type=str)
    parse.add_argument("--backbones", dest="backbones", help="diff backbone get feature",
                       default="mobileNetV2", type=str)
    parse.add_argument("--is_fintuning", dest="is_fintuning",help="whether fintuning",
                       default=False, type=bool)
    parse.add_argument("--st_epoch", dest="st_epoch", help="start epoch of training",
                       default=0, type=int)
    parse.add_argument("--end_epoch", dest="end_epoch", help="end epoch of training",
                       default=500, type=int)
    parse.add_argument("--data_root", dest="data_root", help="dataSet directory",
                       default="./data/cifar10", type=str)
    parse.add_argument("--is_cuda", dest="is_cuda", help="whether use cuda when training net",
                       default=True, type=bool)
    parse.add_argument("--frequent", dest="frequent", help="frequent of printing log",
                       default=10, type=int)
    parse.add_argument("--optim", dest="optim", help="optimizer of training model",
                       default="SGD", type=str)
    
    parse.add_argument("--transform", dest="transform", help="to image transform operator",
                       default=True, type=bool)
    
    args = parse.parse_args()
    return args 
    
if __name__=="__main__":
    
    args = parse_args()
    dataSet_train = genData.genCifar10(args.data_root, is_train=args.is_train, 
                                       transform=args.transform).dataLoader()
    dataSet_valid = genData.genCifar10(args.data_root, is_train=not args.is_train, 
                                       transform=args.transform).dataLoader()

    print("start train " + args.backbones + " model........")
    print("Train total num:", len(dataSet_train))
    print("Valid total num:", len(dataSet_valid)) 
    
    train_model(batch_size=args.batch_size, lr=args.lr, dataSet_train=dataSet_train,
                dataSet_valid=dataSet_valid, 
                ckpt_path=args.ckpt_path, 
                backbones=args.backbones, 
                st_epoch=args.st_epoch, 
                end_epoch=args.end_epoch,
                is_fintuning=args.is_fintuning,
                is_cuda=args.is_cuda,
                frequent=args.frequent,
                optim=args.optim)  
