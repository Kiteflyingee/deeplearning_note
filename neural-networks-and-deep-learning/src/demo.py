#coding=utf-8
'''
Created on 2018年5月7日

@author: devkite
'''
import mnist_loader
import network

trainDataset,validationDataset,testDataset = mnist_loader.load_data_wrapper()
# 训练集是一个50000长度的list，每个元素是一个元祖(x,y),x表示输入特征(768),y表示所属数字label
print(len(trainDataset))
print(len(trainDataset[0]))
# x的结构
print(trainDataset[0][0].shape)
# y的结构
print(trainDataset[0][1].shape)