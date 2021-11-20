"""
这里是对数据集进行预处理的代码
"""
import numpy as np
import pandas as pd
from pandas import to_datetime
from sklearn.decomposition import PCA


way = {"拖网":0,"围网":1,"刺网":2}


def read_csv(csv_path, train):
	df = pd.read_csv(csv_path)
	ID = df.loc[1,"渔船ID"]
	data = df.loc[0:,"lat":"time"]
	target = []
	if train:
		target = way[df.loc[1,"type"]]
	#时间转换
	time =to_datetime(data.loc[:,"time"],format="%Y/%m/%d %H:%M:%S",errors='coerce')
	#由于以一天作为一个数据单位,故将时间设为在一年中第几天+时间,后续可以改成加上年份的！
	data.loc[:,"time"] = time.dt.dayofyear + time.dt.hour/24
	#按天分组并聚合
	groups = data.groupby("time",axis=0)
	mean = groups.agg("mean")
	data = mean.reset_index()
	
	#数据展平
	data = pd.concat([data.iloc[i,:] for i in range(len(data)) ],axis = 0,ignore_index=True)
	if train:
		return ID,data,target
	else:
		return ID,data

# 数据集预处理
def preProcessTrain(dst_path):
	"""
	maxlen = 0
	minlen = 1590
	maxID = 0
	minID = 0
	"""
	train_data = pd.DataFrame()
	train_target =[]
	for i in range(1,18330):
		src_Id,src_data,src_target = read_csv("train_dataset\\train\\{}.csv".format(i),True)
		src_data.to_frame()
		#间隔一定长度，分割数据
		#依靠这里降低数据量，将每条船的数据都变为1000维，如果优化要，修改数据维度的话，就在这里改,对应的下面测试集的处理方式也得改
		if len(src_data) > 1000:
			index = []
			#这里为什么要除以200，一是因为每条数据包括经度、纬度、速度、方向和时间，共5项数据
			#要变成1000列，就需要200条数据结合，这里是等间距取样
			offset = int((len(src_data)-5)/200)
			i = 0
			for p in range(200):
				index = index + list(range(i,i+5))
				i += offset
			src_data = src_data.iloc[index]
			src_data.index = range(1000)
			
		train_data = train_data.append(src_data,ignore_index=True)
		train_target.append(src_target)
		
		train_data.fillna(0, inplace=True)
		print("Processing train_dataset\\train\{}.csv ---> {}".format(src_Id,train_data.shape))
	
	print("打标签中...")
	train_data[1000] = train_target  #修改了维度的话，这里也要改
	print(train_data)
	train_data.to_csv(dst_path,index=False,header = False)

def preProcessTest(dst_path):
	test_data = pd.DataFrame()
	#这里仍然决定不保留ID，因为是连续的
	for i in range(18330,22364):
		src_Id,src_data = read_csv("test_dataset\\{}.csv".format(i),False)
		src_data.to_frame()
		
		if len(src_data) > 1000:
			index = []
			offset = int((len(src_data)-5)/200)
			i = 0
			for p in range(200):
				index = index + list(range(i,i+5))
				i += offset
			src_data = src_data.iloc[index]
			src_data.index = range(1000)
			
		test_data  = test_data.append(src_data,ignore_index=True)
		test_data.fillna(0, inplace=True)
		
		print("Processing test_dataset\\{}.csv".format(src_Id))
	print(test_data)
	test_data.to_csv(dst_path,index=False,header = False)

"""
将预处理之后的数据放到trainSrc2.csv和testSrc2.csv内，可以更改名字，保存到新的csv文件
因为跑一次预处理时间比较长，如果调试参数时，想优化预处理的数据，最好更改一下保存的文件名
"""
#preProcessTrain("trainSrc2.csv")
#preProcessTest("testSrc2.csv")
