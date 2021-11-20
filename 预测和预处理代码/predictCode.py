import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
#这个是将卷积核大小改变之后的版本，训练得比较慢，但是准确率是相对单调递增的
class myCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 15)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv11 = nn.Conv1d(64, 64, 15, padding=7)
        self.bn11 = nn.BatchNorm1d(64)
        self.conv12 = nn.Conv1d(64, 64, 15, padding=7)
        self.bn12 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(15, 7)

        self.conv21 = nn.Conv1d(64, 64, 15, padding=7)
        self.bn21 = nn.BatchNorm1d(64)
        self.conv22 = nn.Conv1d(64, 64, 15, padding=7)
        self.bn22 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(15, 7)
	"""
        self.conv31 = nn.Conv1d(64, 64, 15, padding=7)
        self.bn31 = nn.BatchNorm1d(64)
        self.conv32 = nn.Conv1d(64, 64, 15, padding=7)
        self.bn32 = nn.BatchNorm1d(64)
        self.maxpool3 = nn.MaxPool1d(15, 7)
        
        self.conv41 = nn.Conv1d(64, 64, 15, padding=7)
        self.bn41 = nn.BatchNorm1d(64)
        self.conv42 = nn.Conv1d(64, 64, 15, padding=7)
        self.bn42 = nn.BatchNorm1d(64)
        self.maxpool4 = nn.MaxPool1d(15, 7)
        """
        self.fc1 = nn.Linear(1152, 288)
        self.fc2 = nn.Linear(288, 96)
        self.fc3 = nn.Linear(96, 32)
        self.fc4 = nn.Linear(32, 3)
        

    def forward(self, x):
        x = x.view(-1, 1, 1000)
        c = self.bn1(self.conv1(x))
        x = F.relu(self.bn11(self.conv11(c)))
        
        x = self.bn12(self.conv12(x))
        x += c
        x = F.relu(x)
        m1 = self.maxpool1(x)
        
        x = F.relu(self.bn21(self.conv21(m1)))
       
        x = self.bn22(self.conv22(x))
       
        x += m1
        x = F.relu(x)
        m2 = self.maxpool2(x)
        """
        x = F.relu(self.bn31(self.conv31(m2)))
        x = self.bn32(self.conv32(x))

        x += m2
        x = F.relu(x)
        m3 = self.maxpool3(x)
        
        
        x = F.relu(self.bn41(self.conv41(m3)))
        x = self.bn42(self.conv42(x))
        x += m3
        x = F.relu(x)
        m4 = self.maxpool4(x)
        """
        #print(m2.shape)
        x = m2.view(-1, 1152)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x) 
        
        return x

def get_data(path):
    data = pd.read_csv(path,header=None)
    #读取进来的是字符串，转换为浮点数
    return data.values.astype(np.float32)
    
def splitSampleinRandom(all_samples,trainP):
    All = len(all_samples)
    train_num = int(All*trainP)
    
    shuffled_index = torch.randperm(All)
    train_index = shuffled_index[:train_num]
    valid_index = shuffled_index[train_num:]
    
    train_samples = all_samples[train_index]
    valid_samples = all_samples[valid_index]
    return train_samples,valid_samples

def calc_forward(model, x, is_train):
    with torch.set_grad_enabled(is_train):
        y = model(x)
    return y

#由于每次训练都保存模型，故这里将训练和测试函数分开
def train_valid():
    train_data = get_data("trainSrc2.csv")
    #转换成tensor数据类型
    train_data = torch.from_numpy(train_data).cuda() 
    #分隔训练样本和验证样本
    train_samples, valid_samples = splitSampleinRandom(train_data,0.8)
    #分隔数据以及标签
    train_samples_data = train_samples[:,:1000]
    train_samples_target = train_samples[:,-1].long()
    valid_samples_data = valid_samples[:,:1000]
    valid_samples_target = valid_samples[:,-1].long()
    #print("train_samples_data",train_samples_data.shape)
    #print("train_samples_target",train_samples_target.shape)
    #生成训练集和验证集
    train_set = TensorDataset(train_samples_data, train_samples_target)
    valid_set = TensorDataset(valid_samples_data, valid_samples_target)
    #加载数据
    loader_train = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
    loader_valid = DataLoader(dataset=valid_set, batch_size=16, shuffle=False)
    # 判断是否存在GPU设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using gpu: %s ' % torch.cuda.is_available())
    # 网络放到GPU上
    net = myCNN1D().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    for epoch in range(11):  # 重复多轮训练
        for i, (inputs, labels) in enumerate(loader_train):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 正向传播 +　反向传播 + 优化 
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 输出统计信息
            if i % 100 == 0:
                print('Epoch: %d Minibatch: %5d loss: %.3f' %(epoch + 1, i + 1, loss.item()))
        
        
        net.eval()
        loss_sum = 0.0
        correct = 0
        for i, data in enumerate(loader_valid):
            data,labels  = data
            outputs = calc_forward(net, data, is_train=False)
            loss = criterion(outputs, labels)
            
            loss_sum += loss.item()
            pred = outputs.data.max(1, keepdim=True)[1]                                            
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
        
        accuracy = 100. * correct / len(loader_valid.dataset)
        print('Epoch #{0} --- Validation Loss: {1:.3f} --- Accuracy : {2:.2f}%'.format(epoch, loss_sum / (i + 1),accuracy))
    
    PATH = "./model/model1.pt"
    torch.save(net,PATH)
#这个是对测试集进行预测的代码
def result_model(model):
    model.eval()
    test_data_get = get_data("testSrc2.csv")
    #转换成tensor数据类型
    test_data = torch.from_numpy(test_data_get).cuda()
    #加载数据
    loader_test = DataLoader(dataset = test_data, batch_size=16, shuffle=False)
   
    # 判断是否存在GPU设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using gpu: %s ' % torch.cuda.is_available())
    
    predictions=np.zeros((len(test_data),2),dtype='int')
    k = 0
    q = 18330
    for i, inputs in enumerate(loader_test):
        inputs = inputs.to(device)
        outputs = model(inputs)         
        #_表示的就是具体的value，preds表示下标，1表示在行上操作取最大值，返回类别
        _,preds = torch.max(outputs.data,1)
        ptype = preds.to('cpu').numpy()
        predictions[k:k+len(inputs),1] = preds.to('cpu').numpy()
        predictions[k:k++len(inputs),0] = np.linspace(q,q++len(inputs)-1,len(inputs))
        #可在过程中看到部分结果
        #print(predictions[i:i+len(inputs),:])
        q += len(inputs)
        k += len(inputs)
        
        print('creating: No. ', i, ' process ... total: ',len(test_data))        
    return predictions



if __name__ == '__main__':
    way = np.array(["拖网","围网","刺网"])
    train_valid()
    mymodel = torch.load("./model/model1.pt") 
    pred = result_model(mymodel)
    result = pd.DataFrame(pred,columns= ["渔船ID","type"])
    result.iloc[:,1] = way[pred[:,1]] 
    result.to_csv("result.csv",index=False)
    

    
    
