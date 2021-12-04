import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#网络结构
class myCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 7)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv11 = nn.Conv1d(64, 64, 7, padding=3)
        self.bn11 = nn.BatchNorm1d(64)
        self.conv12 = nn.Conv1d(64, 64, 7, padding=3)
        self.bn12 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(7, 3)

        self.conv21 = nn.Conv1d(64, 64, 7, padding=3)
        self.bn21 = nn.BatchNorm1d(64)
        self.conv22 = nn.Conv1d(64, 64, 7, padding=3)
        self.bn22 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(7, 3)
        
        self.conv31 = nn.Conv1d(64, 64, 7, padding=3)
        self.bn31 = nn.BatchNorm1d(64)
        self.conv32 = nn.Conv1d(64, 64, 7, padding=3)
        self.bn32 = nn.BatchNorm1d(64)
        self.maxpool3 = nn.MaxPool1d(7, 3)
        
        self.conv41 = nn.Conv1d(64, 64, 7, padding=3)
        self.bn41 = nn.BatchNorm1d(64)
        self.conv42 = nn.Conv1d(64, 64, 7, padding=3)
        self.bn42 = nn.BatchNorm1d(64)
        self.maxpool4 = nn.MaxPool1d(7, 3)
        
        self.conv51 = nn.Conv1d(64, 64, 7, padding=3)
        self.bn51 = nn.BatchNorm1d(64)
        self.conv52 = nn.Conv1d(64, 64, 7, padding=3)
        self.bn52 = nn.BatchNorm1d(64)
        self.maxpool5 = nn.MaxPool1d(7, 3)
        """
        self.fc1 = nn.Linear(1888,960)
        self.fc2 = nn.Linear(960, 320)
        self.fc3 = nn.Linear(320, 108)
        self.fc4 = nn.Linear(108, 36)
        self.fc5 = nn.Linear(36, 12)
        self.fc6 = nn.Linear(12, 3)
        """
        self.fc1 = nn.Linear(384,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64 ,32)
        self.fc4 = nn.Linear(32, 8)
        self.fc5 = nn.Linear(8, 3)
        
    def forward(self, x):
        x = x.view(-1, 1, 2000)
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
        
        x = F.relu(self.bn51(self.conv51(m4)))
        x = self.bn52(self.conv52(x))
        x += m4
        x = F.relu(x)
        m5 = self.maxpool5(x)
        
        #print(m5.shape)
        x = m5.view(-1, 384)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        #x = F.relu(self.fc6(x))  
        return x

