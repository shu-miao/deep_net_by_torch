'''
U-net
1D U-net是一种专为处理一维序列数据（音频、时间序列、传感器信号）设计的深度学习模型。
它通过“编码-解码”结构和“跳跃连接”实现高效特征提取与细节恢复。
广泛应用于信号去噪、时序预测、语音增强等任务。
'''

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    '''双层卷积块'''
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),

            # 输入通道是上一层的输出通道数，在双层卷积块中，第二个卷积层的输入通道数和输出通道数都等于第一个卷积层的输出通道数
            nn.Conv1d(out_channels,out_channels,kernel_size=3,padding=1),

            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)


class UNet1D(nn.Module):
    def __init__(self,input_channels=1,output_channels=1):
        super().__init__()
        # 编码器
        self.enc1 = DoubleConv(input_channels,64)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool1d(2)

        # 瓶颈层
        self.bottleneck = DoubleConv(128,256)

        # 解码器
        self.up1 = nn.ConvTranspose1d(256,128,kernel_size=2,stride=2)
        self.dec1 = DoubleConv(256,128)
        self.up2 = nn.ConvTranspose1d(128,64,kernel_size=2,stride=2)
        self.dec2 = DoubleConv(128,64)

        # 输出层
        self.out = nn.Conv1d(64,output_channels,kernel_size=1)

    def forward(self,x):
        # 编码器
        enc1 = self.enc1(x) # 输入形状：[N,1,100]
        enc2 = self.enc2(self.pool1(enc1))

        # 瓶颈层
        bottleneck = self.bottleneck(self.pool2(enc2)) # 形状 [N,256,25]

        # 解码器
        dec1 = self.up1(bottleneck) # 形状 [N,128,50]
        dec1 = torch.cat([dec1,enc2],dim=1) # 拼接后 [N,256,50]
        dec1 = self.dec1(dec1)

        dec2 = self.up2(dec1) # 形状 [N,64,100]
        dec2 = torch.cat([dec2,enc1],dim=1) # 拼接后 [N,128,100]
        dec2 = self.dec2(dec2)

        return self.out(dec2) # 输出形状 [N,1,100]


# 初始化模型
model = UNet1D(input_channels=1,output_channels=1)

# 模拟输入数据：batch_size=4，通道=1，序列长度100
input_tensor = torch.randn(4,1,100)
output = model(input_tensor)
print(output.shape) # 输出形状：torch.Size([4,1,100])