import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# 定义位置编码函数
def positional_encoding(lat, lon, d_model):
    """
    将经纬度转换为位置编码
    :param lat: 纬度
    :param lon: 经度
    :param d_model: 模型维度
    :return: 位置编码
    """
    # 归一化经纬度
    lat = (lat + 90) / 180  # 将纬度转换到[0, 1]范围
    lon = (lon + 180) / 360  # 将经度转换到[0, 1]范围

    # 生成位置编码
    position = np.zeros((1, d_model))
    for pos in range(d_model // 2):
        position[0, 2 * pos] = np.sin(lat * (2 * np.pi) * (1 / (10000 ** (2 * pos / d_model))))
        position[0, 2 * pos + 1] = np.cos(lat * (2 * np.pi) * (1 / (10000 ** (2 * pos / d_model))))
        position[0, 2 * pos] += np.sin(lon * (2 * np.pi) * (1 / (10000 ** (2 * pos / d_model))))
        position[0, 2 * pos + 1] += np.cos(lon * (2 * np.pi) * (1 / (10000 ** (2 * pos / d_model))))

    return torch.tensor(position, dtype=torch.float32)


# 定义Transformer模型
class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, n_heads, num_layers):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, 1)  # 输出速度

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x[-1])  # 取最后一个时间步的输出


# 定义元胞自动机
class CellularAutomaton:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))  # 初始化网格

    def update(self, speed):
        """
        更新元胞状态
        :param speed: 速度值
        """
        # 这里可以根据速度更新元胞状态的逻辑
        self.grid += speed.item()  # 简单示例：将速度加到网格上


# 主函数
def main():
    # 数据载入
    data_file = 'data.csv'  # CSV 文件名
    data = pd.read_csv(data_file)

    # 提取特征和目标速度
    latitudes = data['纬度'].values
    longitudes = data['经度'].values
    features = data[['特征1', '特征2', '特征3', '特征4', '特征5', '特征6', '特征7']].values
    speeds = data['速度'].values

    # 数据预处理
    d_model = 16
    inputs = []
    for lat, lon, *feature in zip(latitudes, longitudes, *features.T):
        pos_enc = positional_encoding(lat, lon, d_model)
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        combined_input = torch.cat((pos_enc, feature_tensor), dim=0)
        inputs.append(combined_input)
    inputs = torch.stack(inputs).unsqueeze(1)  # 增加时间维度

    # 初始化模型
    model = TimeSeriesTransformer(d_model=d_model + 7, n_heads=4, num_layers=2)  # d_model增加特征数量
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 训练模型
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.tensor(speeds, dtype=torch.float32).unsqueeze(1))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

    # 使用模型进行预测
    model.eval()
    predicted_speed = model(inputs[-1].unsqueeze(0))  # 预测最后一个时间步的速度

    # 更新元胞自动机
    ca = CellularAutomaton(grid_size=10)
    ca.update(predicted_speed)

    print("Updated Cellular Automaton Grid:")
    print(ca.grid)


if __name__ == "__main__":
    main()
