'''
kaggle Titanic
二分类任务，预测目标是该乘客是否幸存
'''

# 引入必要的模块
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 首先完成数据集加载
class TitanicDataset(Dataset):
    def __init__(self, path, transform=None):
        self.dataset = pd.read_csv(path)  # 从目录加载数据集 /kaggle/input/titanic/train.csv
        self.transform = transform

        self.dataset['Sex'] = self.dataset['Sex'].map({'male': 0, 'female': 1})
        # 确保所有的特征项都是数值类型
        numeric_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

        for col in numeric_cols:
            self.dataset[col] = pd.to_numeric(self.dataset[col], errors='coerce').fillna(0).astype(np.float32)

        print(self.dataset[numeric_cols].dtypes)

    def __len__(self):
        return len(self.dataset)  # 返回数据集的大小

    def __getitem__(self, idx):
        '''根据索引返回数据项'''
        item = self.dataset.iloc[idx]

        # 取特征和标签
        features = item[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
        label = int(item['Survived'])

        features = np.array(features, dtype=np.float32)
        features = torch.tensor(features, dtype=torch.float32)

        if self.transform:
            features = self.transform(features)
        return features, label


train_path = 'titanic/train.csv'

titanic_train = TitanicDataset(path=train_path)  # 创建数据集实例

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(titanic_train, batch_size=batch_size, shuffle=True)

for features, labels in train_loader:
    print(features, labels)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 2)
        )
        
    def forward(self, x):
        return self.fc_layers(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
net = Net().to(device) # 模型的实例化
print(net)

optim = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)
criterion = nn.CrossEntropyLoss() # 定义交叉熵分类损失函数


num_epochs = 100
best_acc = 0
for epochs in range(num_epochs):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # 训练
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device) # 将数据转移到设备
        optim.zero_grad() # 梯度清零
        outputs = net(features) # 前向传播
        loss = criterion(outputs, labels) # 计算损失
        loss.backward() # 反向传播
        optim.step() # 更新参数
        running_loss += loss.item() # 累加损失
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader) # 计算平均损失
    print(f"Epoch {epochs+1} - Train Acc: {train_acc:.2f}% Loss: {avg_loss:.4f}")
    scheduler.step() # 更新学习率

    if train_acc > best_acc:
        best_acc = train_acc
        torch.save(net.state_dict(), 'best_model.pth')
print('bast Acc:',best_acc)

class TestTitanicDataset(Dataset):
    def __init__(self, path, transform=None):
        '''
        保持和数据集的预处理方式相同
        '''
        self.dataset = pd.read_csv(path)  # 从目录加载数据集 /kaggle/input/titanic/test.csv
        self.transform = transform
        self.dataset.ffill(inplace=True)  # 补全空缺
        self.dataset['Sex'] = self.dataset['Sex'].map({'male': 0, 'female': 1})  # 将性别转换成0，1
        # print(self.dataset['Sex']) # 检查

        # 确保所有的特征项都是数值类型
        numeric_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        for col in numeric_cols:
            self.dataset[col] = pd.to_numeric(self.dataset[col], errors='coerce')
        # 再次补全空缺值
        self.dataset[numeric_cols] = self.dataset[numeric_cols].fillna(0)

    def __len__(self):
        return len(self.dataset)  # 返回数据集的大小

    def __getitem__(self, idx):
        '''根据索引返回数据项'''
        item = self.dataset.iloc[idx]
        # 取特征和标签
        features = item[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
        features = np.array(features, dtype=np.float32)  # 在转换为矩阵之前显式转化为数组
        features = torch.tensor(features, dtype=torch.float32)

        if self.transform:
            features = self.transform(features)
        return features

test_dataset = TestTitanicDataset('titanic/test.csv')
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

# 加载模型
net.load_state_dict(torch.load('best_model.pth', weights_only=True))
net.eval()

# 开始预测
predictions = []
with torch.no_grad():
    for features in test_loader:
        features = features.to(device)
        outputs = net(features)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())

# 生成提交文件
submission = pd.DataFrame({
    'PassengerId': test_dataset.dataset['PassengerId'],
    'Survived': predictions
})
submission.to_csv('titanic/gender_submission.csv', index=False)