import os,time,operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## print(len(tarining))
trainingDigits =r'D:\机器学习1\trainingDigits'
testDigits =r'D:\机器学习1\testDigits'

tarining = (os.listdir(trainingDigits))
test = (os.listdir(testDigits))
def read_file(doc_name):
    data=np.zeros((1,1024))
    f=open(doc_name)
    for i in range(32):
        hang=f.readline()
        for j in range(32):
            data[0,32*i+j]=int(hang[j])
    return data

def dict_list(dic:dict):
    keys = dic.keys()
    values = dic.values()
    lst = [(key,val) for key,val in zip(keys,values)]
    return lst

def xiangsidu(tests,xunlians,labels,k):
    data_hang=xunlians.shape[0]
    zu=np.tile(tests,(data_hang,1))-xunlians
    q=np.sqrt((zu**2).sum(axis=1)).argsort()
    my_dict={}
    for i in range(k):
        votelabel=labels[q[i]]
        my_dict[votelabel]=my_dict.get(votelabel,0)+1
    sortclasscount = sorted(dict_list(my_dict), key=operator.itemgetter(1),reverse=True)

    return sortclasscount[0][0]

def shibie():
    label_list = []
    train_length = len(tarining)
    train_zero = np.zeros((train_length,1024))
    for i in range(train_length):
        doc_name=tarining[i]
        file_label=int(doc_name[0])
        label_list.append(file_label)
        train_zero[i,:]=read_file(r'%s\%s'%(trainingDigits,doc_name))

    errornum = 0
    testnum = len(test)
    errfile = []
    for i in range(testnum):
        testdoc_name=test[i]
        test_label=int(testdoc_name[0])
        testdataor=read_file(r'%s\%s'%(testDigits,testdoc_name))
        result = xiangsidu(testdataor , train_zero , label_list, 3)
        print("正在测试 %d,内容是 %d"%(test_label,result))
        if(result!=test_label):
            errornum+=1
            errfile.append(testdoc_name)
    print("错误数量为: %d"% errornum)
    print("错误的有 :%s"%[i for i in errfile])
    print("准确率为 %.2f%%"%((1-(errornum/float(testnum)))*100))

if __name__ =='__main__':
    a=time.time()
    shibie()
    b=time.time()-a
    print("运行时间为:",b)
