import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

def positional_encoding(lat,lon,d_model):
    '''
    将经纬度转换为位置编码
    :param lat: 纬度
    :param lon: 经度
    :param d_model: 模型维度
    :return: 位置编码
    '''
    # 归一化经纬度
    lat = (lat + 90) / 180
    lon = (lon + 180) / 360

    # 生成位置编码
    position = np.zeros((1,d_model))
    for pos in range(d_model // 2):
        position[0, 2 * pos] = np.sin(lat * (2 * np.pi) * (1 / (10000 ** (2 * pos / d_model))))
        position[0, 2 * pos + 1] = np.cos(lat * (2 * np.pi) * (1 / (10000 ** (2 * pos / d_model))))
        position[0, 2 * pos] += np.sin(lon * (2 * np.pi) * (1 / (10000 ** (2 * pos / d_model))))
        position[0, 2 * pos + 1] += np.cos(lon * (2 * np.pi) * (1 / (10000 ** (2 * pos / d_model))))

    return torch.tensor(position, dtype=torch.float32)
