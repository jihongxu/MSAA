import os
import cv2
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
from torch.utils.data import Dataset


class EyeDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, item):
        image_path, label = self.image_paths[item], self.labels[item]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_paths)


# 划分数据集并写入csv文件
def split_data(save_path, data_img_labels_path, labels_name):
    # 读取csv文件，获取标签信息
    data = pd.read_csv(data_img_labels_path)
    labels = data[labels_name]

    # 创建分层抽样器
    n_splits = 1  # 分成几个fold或者split，默认为10
    test_size = 0.1  # 测试集大小
    random_state = 3407  # 随机种子
    stratified_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    # 进行分层抽样
    for train_val_index, test_index in stratified_split.split(data, labels):
        # 获取训练集和验证集的索引
        train_val_data = data.iloc[train_val_index]
        # 获取测试集的索引
        test_data = data.iloc[test_index]

        # 将训练集和验证集再进行分层抽样，得到训练集和验证集
        train_size = 0.833  # 训练集大小
        stratified_split_train = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size,
                                                        test_size=1 - train_size, random_state=random_state)
        for train_index, val_index in stratified_split_train.split(train_val_data, train_val_data[labels_name]):
            # 获取训练集和验证集
            train_data = train_val_data.iloc[train_index]
            val_data = train_val_data.iloc[val_index]

    train_data.to_csv(save_path + '/train_data.csv', index=False)
    val_data.to_csv(save_path + '/val_data.csv', index=False)
    test_data.to_csv(save_path + '/test_data.csv', index=False)
    print(len(train_data))
    print(len(val_data))
    print(len(test_data))
    print(type(train_data))
    train_counts = train_data['diagnosis'].value_counts()
    print('train_counts:', train_counts)
    val_counts = val_data['diagnosis'].value_counts()
    print('val_counts:', val_counts)
    test_counts = test_data['diagnosis'].value_counts()
    print('test_counts:', test_counts)

    return train_data, val_data, test_data


def get_data(save_path, data_dir):

    train_data = pd.read_csv(save_path + '/train_data.csv')
    val_data = pd.read_csv(save_path + '/val_data.csv')
    test_data = pd.read_csv(save_path + '/test_data.csv')

    train_counts = train_data['diagnosis'].value_counts()
    print('train_counts:')
    print(train_counts)
    val_counts = val_data['diagnosis'].value_counts()
    print('val_counts:')
    print(val_counts)
    test_counts = test_data['diagnosis'].value_counts()
    print('test_counts:')
    print(test_counts)
    train_img_paths = data_dir + '/' + train_data['id_code'].values + '.png'  # 训练数据图片路径
    val_img_paths = data_dir + '/' + val_data['id_code'].values + '.png'  # 验证数据图片路径
    test_img_paths = data_dir + '/' + test_data['id_code'].values + '.png'  # 测试数据图片路径

    train_labels = train_data['diagnosis'].values  # 训练数据图片标签
    val_labels = val_data['diagnosis'].values  # 验证数据图片标签
    test_labels = test_data['diagnosis'].values  # 测试数据图片标签

    return train_img_paths, val_img_paths, test_img_paths, train_labels, val_labels, test_labels


# split_data(r'D:\ProgrammeWorkSpace\PycharmProjects\graduationProject\dataset\aptos2019',
#          r'D:\ProgrammeWorkSpace\PycharmProjects\graduationProject\dataset\aptos2019\train.csv', 'diagnosis')
