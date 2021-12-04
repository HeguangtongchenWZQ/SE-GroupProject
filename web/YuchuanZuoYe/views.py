from django.contrib import messages
from django.shortcuts import render
from django.http import HttpResponseRedirect, Http404, HttpResponse
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from flask import Flask, render_template, request, redirect, url_for, make_response, abort
from werkzeug.routing import BaseConverter
from os import path
from werkzeug.utils import secure_filename
from django.core.files.uploadedfile import InMemoryUploadedFile
from .models import Classfication
import pandas as pd
import numpy as np
from django import forms
from pandas import to_datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import myCNN1D

import json
from django.shortcuts import render

description = {"刺网":r'刺网捕捞是将长带形的网列敷设于水域中，使鱼刺入网目或被网衣缠络后加以捕捞的作业方式',
                "围网":'利用长带形或一囊两翼的网具包围鱼群，采用围捕或结合围张、围拖等方式，迫使鱼群集中于取鱼部或网囊进行捕捞的作业方式',
                "拖网":'拖网捕捞，是指用渔船拖曳囊袋形网具进行捕捞的作业方式'
               }
urls={
       "刺网":'https://baike.baidu.com/item/%E5%88%BA%E7%BD%91%E6%8D%95%E6%8D%9E/9834336',
       "围网":'https://baike.baidu.com/item/%E5%9B%B4%E7%BD%91/419544',
       "拖网":'https://baike.baidu.com/item/%E6%8B%96%E7%BD%91%E6%8D%95%E6%8D%9E/478657'
}
class FileForm(forms.Form):
    ExcelFile = forms.FileField()

# Create your views here.
def index(request):
    """渔船作业识别的主页"""
    return render(request,'YuchuanZuoYe/index.html')
    
# 结果页面1
def classfication1(request):
    """渔船作业识别的低级版"""

    try:
        if request.method=='POST':
            obj = request.FILES.get('test',None)
            print("===============读取数据中===================")
            data = pd.read_csv(obj)
            #print(data)
            print(preProcess(data))
            print(getPosition(data))
            #返回的结果
            result = predict(preProcess(data))

            print('结果是：',result)
            #print("===============保存数据中===================")
            #data.to_csv(obj.name)
            #result返回的是一个列表，render
            return render(request, 'YuchuanZuoYe/classfication1.html',
                          {'result': result[0],'data':description[result[0]],'urls':urls[result[0]]} )
            #return HttpResponseRedirect(reverse('YuchuanZuoYe:classfication1'))
        return render_template('classfication1.html')
    except Exception as e:
        print(e)
    return render(request,'YuchuanZuoYe/classfication1.html')
@login_required
# 结果页面2
def classfication2(request):
    """渔船作业识别的高级版"""
    try:
        if request.method=='POST':
            obj = request.FILES.get('test',None)
            print("===============读取数据中===================")
            data = pd.read_csv(obj)

            result = predict(preProcess(data))
            print('结果是：', result)
            return render(request, 'YuchuanZuoYe/classfication2.html',
                          {'result': result[0], 'data': description[result[0]], 'urls': urls[result[0]]})
            #print(data)
            #print("===============保存数据中===================")
            #data.to_csv(obj.name)
            #return HttpResponseRedirect(reverse('YuchuanZuoYe:classfication2'))
        return render_template('classfication2.html')
    except Exception as e:
        print(e)
    return render(request,'YuchuanZuoYe/classfication2.html')


"""
利用上传的文件（已经转化为uplpad_data即pandas的DataFrame数据）,
利用已有的模型实现预测，返回预测结果
"""
def predict(upload_data):
    upload_data = upload_data.values.astype(np.float32)
    mymodel = torch.load("./model/model1.pt",map_location=torch.device('cpu'))
    way = np.array(["拖网","围网","刺网"]) 
    mymodel.eval()
    #转换成tensor数据类型
    test_data = torch.from_numpy(upload_data)
    #加载数据
    loader_test = DataLoader(dataset = test_data, batch_size=1, shuffle=False)
   
    # 判断是否存在GPU设备
    device = torch.device("cpu")
    
    for i, inputs in enumerate(loader_test):
        inputs = inputs.to(device)
        outputs = mymodel(inputs)         
        #_表示的就是具体的value，preds表示下标，1表示在行上操作取最大值，返回类别
        _,preds = torch.max(outputs.data,1)
        ptype = preds.to('cpu').numpy()
        return way[ptype]

"""
利用上传的文件（已经转化为uplpad_data即pandas的DataFrame数据）
提取出每一天每个小时的数据，返回每天每个小时的平均的经纬度信息（DataFrame格式）
"""
def getPosition(upload_data):
    df = upload_data
    ID = df.loc[1, "渔船ID"]
    data = df.loc[0:, "lat":"time"]
    target = []
    # 时间转换
    time = to_datetime(data.loc[:, "time"], format="%Y/%m/%d %H:%M:%S", errors='coerce')
    # 由于以一个小时作为一个数据单位,故将时间设为在一年中第几天+时间,后续可以改成加上年份的！
    data.loc[:, "time"] = time.dt.dayofyear + time.dt.hour / 24
    # 按天分组并聚合
    groups = data.groupby("time", axis=0)
    mean = groups.agg("mean")
    data = mean.reset_index()
    return data
"""
利用上传的文件（已经转化为uplpad_data即pandas的DataFrame数据）
返回一行数据，2000列(也是DataFrame格式)，preProcess的代码已经实现了大部分功能
"""
def preProcess(upload_data):
    df = upload_data
    ID = df.loc[1, "渔船ID"]
    data = df.loc[0:, "lat":"time"]
    target = []
    # 时间转换
    time = to_datetime(data.loc[:, "time"], format="%Y/%m/%d %H:%M:%S", errors='coerce')
    # 由于以一个小时作为一个数据单位,故将时间设为在一年中第几天+时间,后续可以改成加上年份的！
    data.loc[:, "time"] = time.dt.dayofyear + time.dt.hour / 24
    # 按天分组并聚合
    groups = data.groupby("time", axis=0)
    mean = groups.agg("mean")
    data = mean.reset_index()

    # 数据展平
    data = pd.concat([data.iloc[i, :] for i in range(len(data))], axis=0, ignore_index=True)
    
    test_data = pd.DataFrame(np.zeros((1,2000)))
    # 这里可以不保留ID，
    src_Id, src_data = ID,data
    src_data.to_frame()
    if len(src_data) > 2000:
        index = []
        offset = int((len(src_data) - 5) / 400)
        i = 0
        for p in range(400):
            index = index + list(range(i, i + 5))
            i += offset
        src_data = src_data.iloc[index]
        src_data.index = range(2000)

    test_data.iloc[0,0:len(src_data.index)] = src_data
    test_data.fillna(0, inplace=True)
    return test_data
