'''
如果借助Torch框架，构建模型的过程将会变得十分便捷
example1使用torch框架搭建一个图像分类模型，并使用mnist_images数据集进行训练和测试
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import cv2
from torchviz import make_dot
from torch.utils.data import Dataset, DataLoader

class Net(nn.Module):
    # Net网络类继承自torch.nn的Module类，在构建Net类时应该重写初始化方法和前向传播方法，使用torch搭建网络时，不需要重写反向传播方法，框架中的自动微分机制会自行处理反省传播，只需要在训练过程使用.backward()调用即可
    def __init__(self):
        super(Net,self).__init__() # 调用父类的构造函数，初始化nn.Module类的所有属性和方法
        # torch.nn.Conv2d快速地创建一个卷积层，in_channels:输入通道，out_channels:输出通道，kernel_size:卷积核大小
        self.conv1 = nn.Conv2d(1,6,5) # 1个输入特征数，输出6个特征数，卷积核为5x5
        self.conv2 = nn.Conv2d(6,16,5) # 6个输入特征数，输出16个特征数，卷积核为5x5
        # torch.nn.Linear快速地创建一个全连接层，in_features:输入神经元，out_features:输出神经元
        self.fc1 = nn.Linear(16 * 4 * 4,120) # 输入16 * 4 * 4个特征数，输出为120个特征数
        self.fc2 = nn.Linear(120,84) # 输入120个特征数，输出为84个特征数
        self.fc3 = nn.Linear(84,10) # 输入为84个特征数，输出10个特征数
    def forward(self,input):
        # 前向传播
        # 使用torch.nn.functional可以直接使用某些激活函数
        c1 = F.relu(self.conv1(input)) #进入第一个卷积层，输入为一个通道的input（如灰度图），输出为6个通道的特征图，使用ReLU激活函数
        s2 = F.max_pool2d(c1,2) # 最大池化，池化核大小为2x2，池化层没有可以更新的参数，输出为c1层输出的1/2倍大小的特征图，且通道数一致
        c3 = F.relu(self.conv2(s2)) # 第二个卷积层，输入为s2输出的6个通道的特征图，输出为16个通道的特征图
        s4 = F.max_pool2d(c3,2) # 最大池化层，核大小为2x2，输出为c3层输出的1/2倍大小的特征图，且通道数一致
        s4 = torch.flatten(s4,1) # torch.flatten用于展开特征图张量，它将输入张量展平为一个一维张量，s4:输入为s4,1:从第一个维度开始展平，也就是从行开始
        f5 = F.relu(self.fc1(s4)) # 展平后进入全连接层，输出形状为（图像个数，120）的张量，并使用ReLU激活
        f6 = F.relu(self.fc2(f5)) # 进入第二层全连接，输出为形状（图像个数，84）的张量，并使用ReLU激活
        output = self.fc3(f6) # 最终输出形状为（图像个数，10）的张量
        return output

# 检查可用的Gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Net().to(device)
# print(net) # 查看模型结构
'''
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
'''
params = list(net.parameters())     # 查看模型的可训练参数，nn.parameter()返回参数
# print(len(params))
# print(params[0].size())  # 卷积层1的参数
'''
10
torch.Size([6, 1, 5, 5])
'''

# 加载fashion_mnist_images数据集
def load_mnist_dataset(dataset,path):
    labels = os.listdir(os.path.join(path,dataset))
    X = []
    y = []
    for label in labels:
        for file in os.listdir(os.path.join(path,dataset,label)):
            image = cv2.imread(os.path.join(path,dataset,label,file),cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    return np.array(X),np.array(y).astype('uint8') # 返回数据列表，和标签列表（标签设置为整数）
# 创建一个数据返回函数，创建和返回训练集和测试集
def create_data_mnist(path):
    X,y = load_mnist_dataset('train',path)
    X_test,y_test = load_mnist_dataset('test',path)
    return X,y,X_test,y_test

# 创建数据集
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# 打乱
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# 数据预处理
X = (X.reshape(X.shape[0], 1, 28, 28).astype(np.float32) - 127.5) / 127.5  # 归一化并调整形状
X_test = (X_test.reshape(X_test.shape[0], 1, 28, 28).astype(np.float32) - 127.5) / 127.5

# 转换为Tensor
X_tensor = torch.tensor(X).to(device)
y_tensor = torch.tensor(y).to(device)
X_test_tensor = torch.tensor(X_test).to(device)
y_test_tensor = torch.tensor(y_test).to(device)

# 创建模型
net = Net().to(device)

# 加载优化器
optim = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失

# # 训练和验证
# num_epochs = 1000
# for epoch in range(num_epochs):
#     optim.zero_grad()  # 梯度归零
#     prediction = net(X_tensor)  # 前向传播
#     loss = criterion(prediction, y_tensor)  # 计算损失
#     loss.backward()  # 反向传播
#     optim.step()  # 更新参数
#
#     # 验证
#     with torch.no_grad():  # 禁用梯度计算以提高性能
#         val_prediction = net(X_test_tensor)  # 在测试集上进行前向传播
#         _, predicted = torch.max(val_prediction, 1)  # 获取预测结果
#         correct = (predicted == y_test_tensor).sum().item()  # 计算正确预测的数量
#         accuracy = correct / y_test_tensor.size(0)  # 计算准确率
#
#     # 打印损失、准确率和学习率
#     if epoch % 10 == 0:
#         current_lr = optim.param_groups[0]['lr']  # 获取当前学习率
#         print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}, Learning Rate: {current_lr}')
#
# # 保存模型
# torch.save(net.state_dict(), 'fashion_mnist_model.pth')
# print("模型已保存.")

# 预处理输入图片
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
    image = cv2.resize(image, (28, 28))  # 调整大小为28x28
    image = (image.astype(np.float32) - 127.5) / 127.5  # 归一化
    image = image.reshape(1, 1, 28, 28)  # 调整形状为(1, 1, 28, 28)
    return torch.tensor(image).to(device)



# # 加载模型
# loaded_net = Net().to(device)
# loaded_net.load_state_dict(torch.load('fashion_mnist_model.pth'))
# loaded_net.eval()  # 设置为评估模式
# # 加载并预测
# image_path = 'tshirt.png'  # 输入图片路径
# input_tensor = preprocess_image(image_path)  # 预处理图片
#
# fashion_mnist_labels = {
#     0: 'T-shirt/top',
#     1: 'Trouser',
#     2: 'Pullover',
#     3: 'Dress',
#     4: 'Coat',
#     5: 'Sandal',
#     6: 'Shirt',
#     7: 'Sneaker',
#     8: 'Bag',
#     9: 'Ankle boot'
#     }
#
# with torch.no_grad():
#     output = loaded_net(input_tensor)  # 前向传播
#     _, predicted_label = torch.max(output, 1)  # 获取预测结果
#     print(f'预测类别: {fashion_mnist_labels[predicted_label.item()]}')
'''
经过1000轮次的训练，模型却不能准确预测出类别，且训练的最终损失为0.32，这是一个较高的值
考虑调整模型结构和优化器设置
在模型中增加卷积层以捕获更多特征，增加归一化层以加速训练，增加dropout层，在优化器部分引入L2正则化防止过拟合
并在加载数据是使用框架的方法便于引入批次
'''
class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 增加通道数和使用3x3卷积核
        self.bn1 = nn.BatchNorm2d(16)  # 添加Batch Normalization
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 增加卷积层
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)  # 计算输出特征数
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, input):
        c1 = F.relu(self.bn1(self.conv1(input)))
        s2 = F.max_pool2d(c1, 2)
        c3 = F.relu(self.bn2(self.conv2(s2)))
        s4 = F.max_pool2d(c3, 2)
        c5 = F.relu(self.bn3(self.conv3(s4)))
        s6 = F.max_pool2d(c5, 2)
        s6 = torch.flatten(s6, 1)
        f7 = F.relu(self.fc1(s6))
        f7 = self.dropout(f7)  # 应用Dropout
        f8 = F.relu(self.fc2(f7))
        output = self.fc3(f8)
        return output


# 自定义数据集类
class FashionMNISTDataset(Dataset):
    def __init__(self, dataset, path, transform=None):
        self.X, self.y = self.load_mnist_dataset(dataset, path)
        self.transform = transform

    def load_mnist_dataset(self, dataset, path):
        labels = os.listdir(os.path.join(path, dataset))
        X = []
        y = []
        for label in labels:
            for file in os.listdir(os.path.join(path, dataset, label)):
                image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
                X.append(image)
                y.append(int(label))  # 确保标签为整数
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据预处理函数
def preprocess_data(X):
    # 归一化并调整形状
    return (X.reshape(X.shape[0], 1, 28, 28).astype(np.float32) - 127.5) / 127.5

# 创建数据加载器
def create_data_loaders(path, batch_size=32):
    train_dataset = FashionMNISTDataset('train', path)
    test_dataset = FashionMNISTDataset('test', path)

    # 预处理数据
    train_dataset.X = preprocess_data(train_dataset.X)
    test_dataset.X = preprocess_data(test_dataset.X)

    # 转换为Tensor
    train_dataset.X = torch.tensor(train_dataset.X).to(device)
    train_dataset.y = torch.tensor(train_dataset.y).to(device)
    test_dataset.X = torch.tensor(test_dataset.X).to(device)
    test_dataset.y = torch.tensor(test_dataset.y).to(device)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

# 使用调整后的模型重新训练和预测
# 加载数据集
path = 'fashion_mnist_images'
train_loader, test_loader = create_data_loaders(path, batch_size=32)

improvedNet = ImprovedNet().to(device) # 模型实例化

# 加载优化器
optim = torch.optim.Adam(improvedNet.parameters(), lr=1e-3 ,weight_decay=1e-5)  # 初始学习率
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失

# 创建学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=5) # 自行更新学习率

# 训练和验证
num_epochs = 100
for epoch in range(num_epochs):
    improvedNet.train()  # 设置为训练模式
    running_loss = 0.0
    running_val_loss = 0.0

    # 训练
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optim.zero_grad()  # 梯度归零
        output = improvedNet(images)  # 前向传播
        loss = criterion(output, labels)  # 计算损失
        loss.backward()  # 反向传播
        optim.step()  # 更新参数
        running_loss += loss.item()  # 累加损失

    # 验证
    with torch.no_grad():  # 禁用梯度计算以提高性能
        val_prediction = []
        val_labels = []
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = improvedNet(images)  # 在测试集上进行前向传播
            val_loss = criterion(output, labels)
            running_val_loss += val_loss.item()  # 累加验证损失
            val_prediction.append(output)
            val_labels.append(labels)

        val_prediction = torch.cat(val_prediction)
        val_labels = torch.cat(val_labels)
        _, predicted = torch.max(val_prediction, 1)  # 获取预测结果
        correct = (predicted == val_labels).sum().item()  # 计算正确预测的数量
        accuracy = correct / val_labels.size(0)  # 计算准确率

    # 更新学习率调度器
    scheduler.step(running_val_loss / len(test_loader))  # 根据验证损失更新学习率

    # 打印损失、准确率和学习率
    current_lr = scheduler.get_last_lr()[0]  # 获取当前学习率
    print(f'Epoch {epoch}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy}, Learning Rate: {current_lr}')

# 保存模型
torch.save(improvedNet.state_dict(), 'fashion_mnist_improvedModel.pth')
print("模型已保存.")

# 加载模型
loaded_improvedNet = ImprovedNet().to(device)
loaded_improvedNet.load_state_dict(torch.load('fashion_mnist_improvedModel.pth'))
loaded_improvedNet.eval()  # 设置为评估模式
# 加载并预测
image_path = 'tshirt.png'  # 输入图片路径
input_tensor = preprocess_image(image_path)  # 预处理图片

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
    }

with torch.no_grad():
    output = loaded_improvedNet(input_tensor)  # 前向传播
    _, predicted_label = torch.max(output, 1)  # 获取预测结果
    print(f'预测类别: {fashion_mnist_labels[predicted_label.item()]}')