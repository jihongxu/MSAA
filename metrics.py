import os
import json
import random

import torch.nn as nn
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy import interp
from sklearn.preprocessing import label_binarize
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn import metrics
from matplotlib import colors as mcolors
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

random.seed(125)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown',
          'firebrick', 'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen',
          'darkolivegreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen',
          'mediumseagreen', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy',
          'darkblue', 'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple',
          'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'purple',
          'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink']
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x",
           "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
linestyle = ['--', '-.', '-']


def get_line_arg():
    '''
    随机产生一种绘图线型
    '''
    line_arg = {}
    line_arg['color'] = random.choice(colors)
    # line_arg['marker'] = random.choice(markers)
    line_arg['linestyle'] = random.choice(linestyle)
    line_arg['linewidth'] = random.randint(1, 4)
    # line_arg['markersize'] = random.randint(3, 5)
    return line_arg


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库,将输出打印成列表的形式
    """

    def __init__(self, num_classes: int, labels: list, path):
        # 0-No DR
        # 1-1_Mild
        # 2-2_Moderate
        # 3-3_Severe
        # 4-Proliferative DR
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.path = path
        self.labels = labels
        self.img_path = path + '/images'
        self.json_path = path + '/configs'
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.json_path):
            os.makedirs(self.json_path)

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # 计算 准确率（Accuracy， Acc）
        # 混淆矩阵对角线上的元素之和 为分子
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        marcro_Accuracy = sum_TP / np.sum(self.matrix)
        marcro_error_rate = 1 - marcro_Accuracy

        # precision, recall, specificity
        # 精确率， 召回率， 特异度
        table = PrettyTable()
        table.field_names = ["Name", "Accuracy", "Precision", "Recall(TPR)", "Specificity", "FPR", "F1"]

        Precision_sum = 0
        Recall_sum = 0
        TP_sum = 0
        FP_sum = 0
        FN_sum = 0
        TN_sum = 0
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            # 第i行求和 - TP
            FP = np.sum(self.matrix[i, :]) - TP
            # 第i列求和 - TP
            FN = np.sum(self.matrix[:, i]) - TP
            # 所有和 - TP - FP - FN
            TN = np.sum(self.matrix) - TP - FP - FN

            Accuracy = round(((TP + TN) / (TP + TN + FP + FN)), 3) if TP + TN + FP + FN != 0 else 0
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            FPR = round(FP / (FP + TN), 3) if FP + TN != 0 else 0.
            F1 = round(2 * Precision * Recall / (Precision + Recall), 3) if Precision + Recall != 0 else 0.

            Precision_sum += Precision
            Recall_sum += Recall
            TP_sum += TP
            FP_sum += FP
            TN_sum += TN
            FN_sum += FN

            # 宏观 marcro-   微观micro-
            table.add_row([self.labels[i], Accuracy, Precision, Recall, Specificity, FPR, F1])
        print(table)
        print('宏观 marcro- :')
        marcro_Precision = round(Precision_sum / self.num_classes, 3)
        marcro_Recall = round(Recall_sum / self.num_classes, 3)
        marcro_F1 = round(2 * marcro_Precision * marcro_Recall / (marcro_Precision + marcro_Recall), 3)
        print("Accuracy: ", marcro_Accuracy)
        print("Error rate: ", marcro_error_rate)
        print("Precision: ", marcro_Precision)
        print("Recall: ", marcro_Recall)
        print("F1-Score: ", marcro_F1)
        # 针对整个模型的TP FP FN TN
        print('----------这是一条分界线---------')
        print('微观 micro- :')
        micro_Accuracy = round((TP_sum + TN_sum) / (TP_sum + TN_sum + FP_sum + FN_sum),
                               3) if TP_sum + TN_sum + FP_sum + FN_sum != 0 else 0.
        micro_error_rate = 1 - micro_Accuracy
        micro_Precision = round(TP_sum / (TP_sum + FP_sum), 3) if TP_sum + FP_sum != 0 else 0.
        micro_Recall = round(TP_sum / (TP_sum + FN_sum), 3) if TP_sum + FN_sum != 0 else 0.
        micro_F1 = round(2 * micro_Precision * micro_Recall / (micro_Precision + micro_Recall),
                         3) if micro_Precision + micro_Recall != 0 else 0.
        print("Accuracy: ", micro_Accuracy)
        print("Error rate: ", micro_error_rate)
        print("Precision: ", micro_Precision)
        print("Recall: ", micro_Recall)
        print("F1-Score: ", micro_F1)

        with open(self.json_path + '/results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(table.field_names)
            for row in table._rows:
                writer.writerow(row)

        return table, marcro_Accuracy, marcro_error_rate, marcro_Precision, marcro_Recall, marcro_F1, micro_Accuracy, micro_error_rate, micro_Precision, micro_Recall, micro_F1

    # 画出混淆矩阵
    def plot_confusion_matrix(self):

        matrix = self.matrix
        print(matrix)
        # 绘图格式
        plt.figure(figsize=(10, 10))

        plt.imshow(matrix, cmap=plt.cm.Blues)  # 白到蓝的渐变

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=90)  # 旋转45度
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels', fontsize=25, c='r')
        plt.ylabel('Predicted Labels', fontsize=25, c='r')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                # 大于阈值就是白色，否则是黑色
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
        plt.tight_layout()
        plt.savefig('{}'.format(self.img_path) + '/' + 'Confusion_matrix.png', dpi=300)
        # plt.show()

        return matrix

    # 画ROC曲线
    def plot_roc(self, y_label: list, y_pre: list):

        # ROC曲线（多分类）
        # 在多分类的ROC曲线中，会把目标类别看作是正例，而非目标类别的其他所有类别看作是负例，从而造成负例数量过多，
        # 虽然模型准确率低，但由于在ROC曲线中拥有过多的TN，因此AUC比想象中要大
        # 计算每一类的ROC

        # print(y_label)
        # print('-' * 50)
        # print(outputs_list)
        # print('-' * 50)
        binarize_predict = label_binarize(y_label, classes=[i for i in range(self.num_classes)])
        # 读取预测结果
        predict_score = np.array(y_pre)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(binarize_predict[:, i], [socre_i[i] for socre_i in predict_score])
            # fpr[i], tpr[i], _ = roc_curve(y_label, y_pre, pos_label=i)
            # roc_auc[i] = auc(fpr[i], tpr[i])
            roc_auc[self.labels[i]] = auc(fpr[i], tpr[i])

        # print(fpr)
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.num_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.num_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])  # 插值函数

        # Finally average it and compute AUC
        mean_tpr /= self.num_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        # Plot all ROC curves
        lw = 2

        plt.figure(figsize=(12, 8))  # 设置画布的大小和dpi，为了使图片更加清晰
        plt.plot(fpr["macro"], tpr["macro"], **get_line_arg(),
                 label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]))

        for i in range(self.num_classes):
            plt.plot(fpr[i], tpr[i], **get_line_arg(),
                     label='ROC curve of {0} (area = {1:0.2f})'.format(self.labels[i], roc_auc[self.labels[i]]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class receiver operating characteristic ')
        plt.legend(loc="lower right")
        plt.savefig('{}'.format(self.img_path) + '/' + 'ROC_Figure.png', dpi=120, bbox_inches='tight')

        # plt.show()
        return roc_auc

    # 画PR曲线
    def plot_pr1(self, y_label: list, y_pre: list):
        # print(y_label)  # [0, 1, 3, 4, 0 ....]
        # print(y_pre)  # [[1.1, 2.1, 3.3, -1.5, -3.2], [], [].....]

        # 使用label_binarize让数据成为类似多标签的设置
        binarize_predict = label_binarize(y_label, classes=[i for i in range(self.num_classes)])
        # print(binarize_predict)  # [[1 0 0 0 0], [0 1 0 0 0], [].....]

        # 读取预测结果
        predict_score = np.array(y_pre)

        # print(y_pre)  # [[1.1, 2.1, 3.3, -1.5, -3.2], [], [].....]
        precision_dict = dict()
        recall_dict = dict()
        average_precision_dict = dict()
        for i in range(self.num_classes):
            precision_dict[i], recall_dict[i], _ = precision_recall_curve(binarize_predict[:, i],
                                                                          [socre_i[i] for socre_i in predict_score])
            average_precision_dict[self.labels[i]] = average_precision_score(binarize_predict[:, i],
                                                                             [socre_i[i] for socre_i in predict_score])

        average_precision_dict["macro"] = sum(average_precision_dict.values()) / len(average_precision_dict)

        # micro
        precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(binarize_predict.ravel(),
                                                                                  predict_score.ravel())
        average_precision_dict["micro"] = average_precision_score(binarize_predict, predict_score, average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(
            average_precision_dict["micro"]))

        # Plot all PR curves
        lw = 2
        plt.figure(figsize=(12, 14))  # 设置画布的大小和dpi，为了使图片更加清晰
        plt.plot(recall_dict["micro"], precision_dict["micro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(average_precision_dict["micro"]),
                 color='navy', linestyle=':', linewidth=4)

        for i in range(self.num_classes):
            plt.plot(recall_dict[i], precision_dict[i], lw=lw,
                     label='ROC curve of {0} (area = {1:0.2f})'.format(self.labels[i],
                                                                       average_precision_dict[self.labels[i]]))

        # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.plot([0, 0], [0, 1], ls="--", c='.3', linewidth=3, label='随机模型')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class ')
        plt.legend(loc="lower left")
        plt.savefig('{}'.format(self.img_path) + '/' + 'PR_Figure.png', dpi=120, bbox_inches='tight')
        # plt.show()

        return average_precision_dict

    # 画PR曲线
    def plot_pr(self, y_label: list, y_pre: list):
        y_label = np.array(y_label)
        y_pre = np.array(y_pre)

        plt.figure(figsize=(14, 10))
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.01])
        # plt.plot([0, 1], [0, 1],ls="--", c='.3', linewidth=3, label='随机模型')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.rcParams['font.size'] = 22
        plt.grid(True)

        # 初始化字典存储每个类别的Precision，Recall和AP
        pr_dict = {}
        ap_dict = {}

        for i in range(self.num_classes):
            # 将y_label中等于当前类别的样本置为1，其余置为0
            y_true = (y_label == i).astype(int)

            # 获取当前类别的预测概率
            y_score = y_pre[:, i]

            # 计算Precision、Recall、阈值
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)

            # 计算AP
            ap = average_precision_score(y_true, y_score)

            # 存储结果到字典
            pr_dict[i] = (precision, recall)
            ap_dict[self.labels[i]] = ap

            # 绘制PR曲线
            plt.plot(recall, precision, **get_line_arg(),
                     label='PR of {0} (AP = {1:0.2f})'.format(self.labels[i], ap))
            plt.legend()

        # 计算所有类别的平均AP
        mAP = np.mean(list(ap_dict.values()))
        ap_dict["AP"] = mAP
        plt.title(f"PR Curve (mAP = {mAP:.3f})")
        plt.legend(loc='best', fontsize=12)
        plt.savefig('{}'.format(self.img_path) + '/' + 'PR_Figure.png', dpi=120, bbox_inches='tight')
        # plt.show()

        return ap_dict

    # 0.0~0.20极低的一致性(slight)、0.21-0.40一般的一致性(fair)、
    # 0.41-0.60 中等的一致性(moderate)、0.61-0.80
    # 高度的一致性(substantial)和0.81-1几乎完全一致(almost perfect)。

    def compute_kappa(self):
        N = np.sum(self.matrix)
        T = np.sum(np.diag(self.matrix))
        p0 = T / N
        pe = np.sum(np.sum(self.matrix, axis=0) * np.sum(self.matrix, axis=1)) / (N * N)
        kappa = (p0 - pe) / (1 - pe)
        # print('N', N)
        # print('T', T)
        # print('p0', p0)
        # print('pe', pe)
        return kappa


# 衡量指标
def quadratic_weighted_kappa(y_pred, y_true):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()
    if y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0]
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return metrics.cohen_kappa_score(y_pred, y_true, weights='quadratic')
