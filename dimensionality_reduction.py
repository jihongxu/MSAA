import os.path

import pandas as pd
import numpy as np
import warnings
import torch
import cv2
import seaborn as sns
# import umap.umap_ as umap
# import umap
# import umap.plot
import random
from sklearn.manifold import TSNE
import plotly.express as px
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

# warnings.filterwarnings("ignore")
marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
class_list = ['No DR', '1_Mild', '2_Moderate', '3_Severe', 'Proliferative DR']


def semantic_features(device, model, model_node_name, test_img_paths, test_transform, model_parameter_path, model_parameter_save_path):

    node_names = get_graph_node_names(model)
    # print(node_names)  # 查看模型中的节点名称

    model.load_state_dict(torch.load(model_parameter_path, map_location=device), False)
    model.to(device)

    # 定义需要返回的节点，这里我们只需要avg_pool的输出结果
    return_nodes = {model_node_name: 'semantic_feature'}
    # 定义特征提取器
    feature_extractor = create_feature_extractor(model, return_nodes)

    # 将特征提取器置为评估模式
    model.eval()
    encoding_array = []
    img_path_list = []
    feature_extractor.eval()
    for img_path in tqdm(test_img_paths):
        img_path_list.append(img_path)
        img_pil = Image.open(img_path).convert('RGB')
        input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
        feature = feature_extractor(input_img)[
            'semantic_feature'].squeeze().detach().cpu().numpy()  # 执行前向预测，得到 avgpool 层输出的语义特征
        encoding_array.append(feature)

    encoding_array = np.array(encoding_array)
    print(encoding_array.shape)

    # 保存为本地的 npy 文件
    np.save(model_parameter_save_path + '/测试集语义特征.npy', encoding_array)


# def umap1(model_parameter_save_path, test_labels):
#     encoding_array = np.load(model_parameter_save_path + '/测试集语义特征.npy', allow_pickle=True)
#     print(encoding_array.shape)
#     name = []
#     for i in test_labels:
#         name.append(class_list[i])
#
#     pd.DataFrame(data={'标注类别名称': name})
#     print(pd)
#
#     # 可视化配置
#     n_class = len(class_list)  # 测试集标签类别数
#     palette = sns.hls_palette(n_class)  # 配色方案
#     sns.palplot(palette)
#     random.seed(1234)
#     random.shuffle(marker_list)
#     random.shuffle(palette)
#
#     mapper = umap.UMAP(n_neighbors=10, n_components=2, random_state=12).fit(encoding_array)
#     print(mapper.embedding_.shape)
#     X_umap_2d = mapper.embedding_
#     print(X_umap_2d.shape)
#     # 不同的 符号 表示 不同的 标注类别
#     show_feature = '标注类别名称'
#     plt.figure(figsize=(14, 14))
#     for idx, name in enumerate(class_list):  # 遍历每个类别
#         # 获取颜色和点型
#         color = palette[idx]
#         marker = marker_list[idx % len(marker_list)]
#
#         # 找到所有标注类别为当前类别的图像索引号
#         indices = np.where(df[show_feature] == name)
#         plt.scatter(X_umap_2d[indices, 0], X_umap_2d[indices, 1], color=color, marker=marker, label=fruit, s=150)
#
#     plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
#     plt.xticks([])
#     plt.yticks([])
#     plt.savefig('语义特征UMAP二维降维可视化.pdf', dpi=300)  # 保存图像
#     plt.show()


def t_SNE1(model_parameter_save_path, test_labels, test_img_paths):

    encoding_array = np.load(model_parameter_save_path + '/测试集语义特征.npy', allow_pickle=True)
    print(encoding_array.shape)

    name_list = []
    for i in test_labels:
        name_list.append(class_list[i])

    img_path_list = []
    for i in test_img_paths:
        img_path_list.append(os.path.basename(i))

    df = pd.DataFrame(data={'标注类别名称': name_list, "图像路径": img_path_list})

    print(df)
    # 可视化配置
    n_class = len(class_list)  # 测试集标签类别数
    palette = sns.hls_palette(n_class)  # 配色方案
    sns.palplot(palette)
    random.seed(1234)
    random.shuffle(marker_list)
    random.shuffle(palette)

    # t-SNE降至到二维
    tsne = TSNE(n_components=2, n_iter=20000)
    X_tsne_2d = tsne.fit_transform(encoding_array)
    # print(X_tsne_2d.shape)  # (367, 2)

    # 不同的 符号 表示 不同的 标注类别
    show_feature = '标注类别名称'

    # 可视化展示
    plt.figure(figsize=(10, 10))
    for idx, name in enumerate(class_list):  # 遍历每个类别
        # 获取颜色和点型
        color = palette[idx]
        marker = marker_list[idx % len(marker_list)]

        # 找到所有标注类别为当前类别的图像索引号
        indices = np.where(df[show_feature] == name)
        plt.scatter(X_tsne_2d[indices, 0], X_tsne_2d[indices, 1], color=color, marker=marker, label=name, s=150)

    plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(model_parameter_save_path + '/../images/语义特征t-SNE二维降维可视化.pdf', dpi=300)  # 保存图像
    plt.show()

    # ploply交互式可视化
    df_2d = pd.DataFrame()
    df_2d['X'] = list(X_tsne_2d[:, 0].squeeze())
    df_2d['Y'] = list(X_tsne_2d[:, 1].squeeze())
    df_2d['标注类别名称'] = df['标注类别名称']
    # df_2d['预测类别'] = df['top-1-预测名称']
    df_2d['图像路径'] = df['图像路径']
    df_2d.to_csv(model_parameter_save_path + '/../configs/t-SNE-2D.csv', index=False)
    print(df_2d)

    fig = px.scatter(df_2d,
                     x='X',
                     y='Y',
                     color=show_feature,
                     labels=show_feature,
                     symbol=show_feature,
                     hover_name='图像路径',
                     opacity=0.8,
                     width=1000,
                     height=600
                     )
    # 设置排版
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
    fig.write_html(model_parameter_save_path + '/../images/语义特征t-SNE二维降维plotly可视化.html')