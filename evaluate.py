import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import torchvision
from copy import deepcopy
from torchvision import datasets, transforms
import json
import pandas as pd
import time
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
import itertools
import sys
from utils import *
from metrics import *


def evaluate(name, num_class, test_loader, model, model_path, test_labels, output_path, device):
    # 加载训练完的参数
    assert os.path.exists(model_path), "can't find {} file!".format(model_path)
    model.load_state_dict(torch.load(model_path, map_location=device), False)
    model.to(device)

    path = output_path + '/' + 'Validation_{}'.format(name)
    confusion = ConfusionMatrix(num_classes=num_class, labels=test_labels, path=path)

    # 验证模式
    model.eval()
    test_list = list()
    predict_list = list()

    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_images, test_labels = test_data
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            outputs = model(test_images)
            print('outputs1....')
            print(outputs)
            predict_list += [score_i for i, score_i in enumerate(outputs.tolist())]
            print('predict....')
            print(predict_list)
            outputs = torch.argmax(outputs, dim=1)
            print('outputs2....')
            print(outputs)
            confusion.update(outputs.to('cpu').numpy(), test_labels.to("cpu").numpy)
            test_list += test_labels.tolist()

    print('test_list....')
    print(test_list)
    print('predict_list')
    print(predict_list)

    matrix = confusion.plot_confusion_matrix()
    roc_auc = confusion.plot_roc(test_list, predict_list)
    average_precision_dict = confusion.plot_pr(test_list, predict_list)
    table, marcro_Accuracy, marcro_error_rate, marcro_Precision, marcro_Recall, marcro_F1, micro_Accuracy, micro_error_rate, micro_Precision, micro_Recall, micro_F1 = confusion.summary()
    kappa = confusion.compute_kappa()
    result = [{
        "kappa": kappa,
        # 混淆矩阵
        # "confusion_matrix": matrix.tolist(),
        # roc
        "rocAuc": roc_auc,
        # map
        "averagePrecisionDict": average_precision_dict,
        # 每个分类的准确率、召回率表格...
        # "table": table,
        "marcroAccuracy": marcro_Accuracy,
        "marcroErrorRate": marcro_error_rate,
        "marcroPrecision": marcro_Precision,
        "marcroRecall": marcro_Recall,
        "marcroF1": marcro_F1,
        "microAccuracy": micro_Accuracy,
        "microErrorRate": micro_error_rate,
        "microPrecision": micro_Precision,
        "microRecall": micro_Recall,
        "microF1": micro_F1,
        "confusionMatrixFigure": "{}Confusion_matrix.png".format(path),
        "RocFigure": "{}ROC_Figure.png".format(path),
        "PrFigure": "{}PR_Figure.png".format(path)
    }]
    return result


def evaluate2(num_class, test_loader, model, model_path, class_name, output_path, device):

    img_path = output_path + '/images'
    # 加载训练完的参数
    assert os.path.exists(model_path), "can't find {} file!".format(model_path)

    model.load_state_dict(torch.load(model_path, map_location=device), False)
    model.to(device)

    confusion = ConfusionMatrix(num_classes=num_class, labels=class_name, path=output_path)


    # 验证模式
    model.eval()
    test_list = []
    predict_list = []

    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_images, test_labels = test_data
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            outputs = model(test_images)
            predict_list += [score_i for i, score_i in enumerate(outputs.cpu().numpy().tolist())]
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to('cpu').numpy(), test_labels.to('cpu').numpy())
            test_list += test_labels.tolist()

    matrix = confusion.plot_confusion_matrix()
    roc_auc = confusion.plot_roc(test_list, predict_list)
    average_precision_dict = confusion.plot_pr(test_list, predict_list)
    table, marcro_Accuracy, marcro_error_rate, marcro_Precision, marcro_Recall, marcro_F1, micro_Accuracy, micro_error_rate, micro_Precision, micro_Recall, micro_F1 = confusion.summary()

    kappa = confusion.compute_kappa()

    result = [{
        "kappa": kappa,
        # 混淆矩阵
        # "confusion_matrix": matrix.tolist(),
        # roc
        "rocAuc": {str(k): float(v) for k, v in roc_auc.items()},
        # map
        "averagePrecisionDict": {str(k): float(v) for k, v in average_precision_dict.items()},
        # 每个分类的准确率、召回率表格...
        # "table": table,
        "marcroAccuracy": marcro_Accuracy,
        "marcroErrorRate": marcro_error_rate,
        "marcroPrecision": marcro_Precision,
        "marcroRecall": marcro_Recall,
        "marcroF1": marcro_F1,
        "microAccuracy": micro_Accuracy,
        "microErrorRate": micro_error_rate,
        "microPrecision": micro_Precision,
        "microRecall": micro_Recall,
        "microF1": micro_F1,
        "confusionMatrixFigure": "{}Confusion_matrix.png".format(img_path),
        "RocFigure": "{}ROC_Figure.png".format(img_path),
        "PrFigure": "{}PR_Figure.png".format(img_path),
    }]
    return result
