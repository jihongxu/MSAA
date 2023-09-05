import adabound
import joblib
import torch.backends.cudnn as cudnn

import sys

#from environment.yy.Lib.email.policy import default

sys.path.append("C:\\Users\\30793\\Desktop\\huyang\\code\\graduationProject")

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from V2.model_utils.dataset import EyeDataset, split_data, get_data
from V2.model_utils.visualization.visualization_DFF import *
from V2.model_utils.visualization.visualization_grad_cam import *
from V2.model_utils.visualization.visualization_lime import *
from losses import *
from optimizers import *
from V2.model_utils.data_processing.preprocess import *
from dataset import *
from train import *
from evaluate import *
from models import *
from dimensionality_reduction import *
from V2.model_utils.visualization.visualization_cam import *
from V2.model_utils.visualization.visualization_shap import *
from V2.model_utils.visualization.visualization_captum import *

#硬件和环境配置。
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_name = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
class_name_dict = {
    0: 'No DR',
    1: '1_Mild',
    2: '2_Moderate',
    3: '3_Severe',
    4: 'Proliferative DR'
}

print(torch.cuda.is_available())
print(device)


# 使用随机化种子使神经网络的可重复性
def seed_everything(seed=1):
    """
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='MobileNetv3',
                        choices=['MobileNetv3', 'MobileNetV3_mish_test', 'MobileNetV3_att_eca',
                                 'MobileNetV3_x7_x10_noAtt', 'MobileNetV3_x7_x10', 'MobileNetV3_att_ca',
                                 'MobileNetV3_att_cbam', 'MobileNetV3_att_A2', 'MobileNetV3_att_eca_mish',
                                 'MobileNetV3_att_NAM', 'resnet50', 'MobileNetv2',
                                 'ShuffNet', 'AlexNetV1', 'AlexNetV2',
                                 'MobileNetv3_x5_x7_mish', 'MobileNetv3_mish', 'MobileNetv3_test',
                                 'MobileNetV3_mish_test', 'MobileNetV3_att_dan', 'MobileNetV3_att_se',
                                 'MobileNetV3_att_cbam2', 'MobileNetV3_att_psa', 'MobileNetV3_att_sge',
                                 'MobileNetV3_att_sk', 'MobileNetV3_att_ccnet',
                                 'MobileNetV3_x7_x10_mish', 'MobileNetV3_x7_x10_noAtt_mish', 'MobileNetv3_att_gam'],
                        help='model name')
    parser.add_argument('--mode', default='train',
                        choices=['train', 'continue_train', 'evaluate', 'semantic_features', 'visualization',
                                 'layer_visualization'])
    parser.add_argument('--visual_method', default='CAM',
                        choices=['CAM', 'Grad_CAM', 'DFF', 'Lime', 'Shap', 'Captum_Integrated_Gradients',
                                 'Captum_GradientShap'])     #可视化操作

    # data source
    parser.add_argument('--dataset', default='aptos2019',
                        help='train dataset')
    parser.add_argument('--num_class', default=5, type=int,
                        help='classify numbers')
    # parser.add_argument('--n_splits', default=5, type=int,
    #                     help='Number of folds for cross verification')

    # preoricessing
    parser.add_argument('--img_size', default=288, type=int,
                        help='input image size(defult: 288)')
    parser.add_argument('--processing', default=False, type=str2bool) #将字符串转换为布尔类型，用于指定是否进行图像预处理。
    parser.add_argument('--scale_radius', default=True, type=str2bool) #是否对图像进行半径缩放
    parser.add_argument('--normalize', default=True, type=str2bool)  #是否对图像进行归一化
    # parser.add_argument('--brightness', default=True, type=str2bool)
    parser.add_argument('--remove', default=True, type=str2bool)      #是否删除无效区域
    parser.add_argument('--input_size', default=256, type=int,
                        help='input image size (default: 256)')

    # data augmentation   一系列的数据增强
    parser.add_argument('--rotate', default=True, type=str2bool)
    parser.add_argument('--rotate_min', default=-180, type=int)
    parser.add_argument('--rotate_max', default=180, type=int)
    parser.add_argument('--rescale', default=True, type=str2bool)
    parser.add_argument('--rescale_min', default=0.8889, type=float)
    parser.add_argument('--rescale_max', default=1.0, type=float)
    parser.add_argument('--shear', default=True, type=str2bool)
    parser.add_argument('--shear_min', default=-36, type=int)
    parser.add_argument('--shear_max', default=36, type=int)
    parser.add_argument('--translate', default=False, type=str2bool)
    parser.add_argument('--translate_min', default=0, type=float)
    parser.add_argument('--translate_max', default=0, type=float)
    parser.add_argument('--flip', default=True, type=str2bool)
    parser.add_argument('--contrast', default=True, type=str2bool)
    parser.add_argument('--contrast_min', default=0.9, type=float)
    parser.add_argument('--contrast_max', default=1.1, type=float)
    parser.add_argument('--random_erase', default=False, type=str2bool)
    parser.add_argument('--random_erase_prob', default=0.5, type=float)
    parser.add_argument('--random_erase_sl', default=0.02, type=float)
    parser.add_argument('--random_erase_sh', default=0.4, type=float)
    parser.add_argument('--random_erase_r', default=0.3, type=float)

    # train
    parser.add_argument('--pretrained', default=True, type=str2bool)             #是否使用预训练模型
    parser.add_argument('--optimizer', default='RAdam',                          #优化器的选择
                        choices=['Adam', 'AdamW', 'RAdam', 'SGD', 'AdaBound'])
    parser.add_argument('--loss', default='FocalLoss',                           #损失函数的选择
                        choices=['FocalLoss', 'MSELoss', 'CrossEntropyLoss'])
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch',  default=4, type=int, metavar='b',
                        help='mini-batch size(default: 32)')
    parser.add_argument('--num_worker', default=0, type=int)                     #工作线程的数量

    # optimizer
    parser.add_argument('--nesterov', default=False, type=str2bool,              #是否使用 Nesterov 加速梯度
                        help='nesterov')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,     #初始学习率，默认为 1e-3
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min_lr', default=1e-7, type=float,                    #最小学习率
                        help='minimum learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,                   #动量，默认为 0.9
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-2, type=float,              #权重衰减，默认为 1e-2
                        help='weight decay')

    # scheduler              学习率调度器
    parser.add_argument('--scheduler', default='ReduceLROnPlateau',              #学习率调度器的选择
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--factor', default=0.5, type=float)                     #ReduceLROnPlateau 调度器的衰减因子，默认为 0.5
    parser.add_argument('--patience', default=5, type=int)                       #ReduceLROnPlateau 调度器的耐心值，默认为 5

    # path
    parser.add_argument('--model_parameter_name', default='best60_0.9492.pth',           #模型评估时的图片路径
                        help='模型评估-模型pth输入路径 ')
    parser.add_argument('--continue_path', default=r'epoch_59.pth')              #继续训练时，上次保存的模型.pth 文件的路径，默认为 'epoch_59.pth
    parser.add_argument('--model_name_suffix', default='V2')                     #模型名称后缀，默认为 'V2'。
    parser.add_argument('--output_path', default=r'C:\Users\30793\Desktop\huyang\code\graduationProject\V2\outputs')
    args = parser.parse_args()                                                   #输出路径
    return args


def saveLocalFile(json_path, json_data):
    # 本地保存
    save_json = json.dumps(json_data, sort_keys=False, indent=2, cls=MyEncoder)
    f = open(json_path, "w")
    f.write(save_json)
    f.close()
# json_data：要转换为 JSON 的数据。
# sort_keys=False：可选参数，指定是否按键排序，默认为 False，表示不排序。
# indent=2：可选参数，指定缩进的空格数，默认为 None，表示不进行缩进。在此例中，设置为 2，表示使用 2 个空格进行缩进。
# cls=MyEncoder：可选参数，指定用于编码的类。MyEncoder 是一个自定义的编码器类，用于处理特定的编码需求。如果不需要自定义编码器，可以省略该参数。
class None89:
    pass


def main():
    seed_everything(seed=45)
    args = arg_parse()

    path = args.output_path + '/'+ args.model_name_suffix + '/' + args.name
    img_path = path + '/images'  # 模型评估-输出的图片路径 1
    json_path = path + '/configs'  # 模型评估-输出的json路径 1
    model_parameter_save_path = path + '/checkpoint'  # 模型训练-训练的模型参数路径
    models_file_path = path  # 模型运行的args相关数据 1
    model_parameter_path = model_parameter_save_path + '/' + args.model_parameter_name
    continue_save_path = model_parameter_save_path + '/state' + '/' + args.continue_path
    # D:\ProgrammeWorkSpace\PycharmProjects\graduationProject\V2\outputs\MobileNetv303_08_18_00\checkpoint\best_0.764.pth
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    if not os.path.exists(model_parameter_save_path):
        os.makedirs(model_parameter_save_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if args.name is None:
        args.name = '%s_%s' % (args.name, datetime.now().strftime('%m%d%H'))

    print('Config --------')
    for arg in vars(args):
        print('- %s: %s' % (arg, getattr(args, arg)))
    print('-----------')

    with open('%s/%s_epochs%s.txt' % (models_file_path, args.name, args.epochs),
              'w') as f:
        for arg in vars(args):
            print('- %s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, '%s/%s_epoch%s_args.pkl' % (models_file_path, args.name, args.epochs))
#该函数的目的是根据命令行参数构建文件路径、创建目录、打印配置信息并保存配置信息到文件。
#具体的实现逻辑可能涉及到其他未提供的函数或类的定义。

    # loss模块    根据 args.loss 的取值不同，选择不同的损失函数。
    losses = args.loss
    if losses == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()
    elif losses == 'FocalLoss':
        criterion = FocalLoss().cuda()
        # 类别加权训练
        # criterion = FocalLoss1(gamma=2, alpha=torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0])).cuda()

    elif losses == 'MSELoss':
        criterion = nn.MSELoss().cuda()
    else:
        raise NotImplementedError

    # model模块
    model = get_model(args.name, args.num_class, pretrained=args.pretrained, device=device)
    # model = mobilenetv3_large_ca(num_classes=args.num_class)
    # model = mobilenetv3_large(num_classes=args.num_class)

    # model = get_mobilenetv3_large(pretrained=False)
    # num_features = model.classifier[-1].in_features
    # model.classifier[-1] = nn.Linear(num_features, args.num_class)

    # 图片数据增强transform模块  112/224/448 =>> 128/256/512
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomAffine(                                                   #进行随机仿射变换，包括旋转、平移、缩放和剪切
            degrees=(args.rotate_min, args.rotate_max) if args.rotate else 0,
            translate=(args.translate_min, args.translate_max) if args.translate else None,
            scale=(args.rescale_min, args.rescale_max) if args.rescale else None,
            shear=(args.shear_min, args.shear_max) if args.shear else None,
        ),
        transforms.CenterCrop(args.input_size),                                   #以图像中心为中心进行裁剪
        transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),               #以 50% 的概率对图像进行水平翻转
        transforms.RandomVerticalFlip(p=0.5 if args.flip else 0),                 #以 50% 的概率对图像进行垂直翻转
        transforms.ColorJitter(                                                   #对图像进行颜色增强处理
            brightness=0,
            contrast=args.contrast,
            saturation=0,
            hue=0),
        RandomErase(                                                             #以一定概率随机擦除图像的一部分区域
            prob=args.random_erase_prob if args.random_erase else 0,
            sl=args.random_erase_sl,
            sh=args.random_erase_sh,
            r=args.random_erase_r),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # optimizer                       训练过程中的优化器（optimizer）
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RAdam':
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == 'AdaBound':
        optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)

    # scheduler           StepLR是什么        训练过程中的学习率调度器（scheduler）
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience,
                                                   verbose=1, min_lr=args.min_lr)
    else:
        scheduler = None89

    # ！！！dataset preprocess
    # dataset_path = r"D:\ProgrammeWorkSpace\PycharmProjects\graduationProject\dataset\aptos2019\eyeDataset"
    dataset_path = r"C:\Users\30793\Desktop\huyang\code\graduationProject\dataset\aptos2019\eyeDataset"
    save_path = r"C:\Users\30793\Desktop\姬红绪\eyepacs"
    csv_path = r"C:\Users\30793\Desktop\姬红绪\eyepacs\trainLabels.csv"
    data_dir, data_df = preprocess(csv_path, dataset_path, save_path, args.dataset, args.img_size,
                                   scale=args.scale_radius, norm=args.normalize, remove=args.remove)

    save_path = r'C:\Users\30793\Desktop\姬红绪\eyepacs'
    if args.processing:
        data_img_labels_path = csv_path
        labels_name = 'diagnosis'
        train_data, val_data, test_data = split_data(save_path, data_img_labels_path, labels_name)

    train_img_paths, val_img_paths, test_img_paths, train_labels, val_labels, test_labels = get_data(save_path,
                                                                                                     data_dir)
    # 10%：test: 916
    # 15%: val: 1377
    # 75%: train: 6867
    print()
    sum = len(train_img_paths) + len(val_img_paths) + len(test_img_paths)
    print('Train Data Size:', len(train_img_paths), ' 占比{:.3f}%'.format(len(train_img_paths) / sum * 100))
    print('Validation Data Size:', len(val_img_paths), ' 占比{:.3f}%'.format(len(val_img_paths) / sum * 100))
    print('Test Data Size:', len(test_img_paths), ' 占比{:.3f}%'.format(len(test_img_paths) / sum * 100))

    train_set = EyeDataset(train_img_paths, train_labels, transform=train_transform)
    val_set = EyeDataset(val_img_paths, val_labels, transform=val_transform)
    test_set = EyeDataset(test_img_paths, test_labels, transform=test_transform)

    # 使用 DataLoader 将数据集转换为可迭代的数据加载器
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.num_worker)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.num_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.num_worker)

    # node_names = get_graph_node_names(model)
    # print(node_names)  # 查看模型中的节点名称
    # return
    # print(torch.cuda.memory_summary())

    if args.mode == 'train':
        # train4(train_loader, val_loader, model, criterion, optimizer, args.epochs, device, args.scheduler,
        #        model_parameter_save_path, scheduler)
        train5(train_loader, val_loader, test_loader, model, criterion, optimizer, args.epochs, device, args.scheduler,
               model_parameter_save_path, scheduler)

    elif args.mode == 'continue_train':
        checkpoint = torch.load(continue_save_path)  # 加载检查点
        print(checkpoint)
        # continue_train(train_loader, val_loader, model, criterion, optimizer, args.epochs, device, args.scheduler,
        #                model_parameter_save_path, checkpoint, scheduler)
        continue_train1(train_loader, val_loader, test_loader, model, criterion, optimizer, args.epochs, device,
                        args.scheduler, model_parameter_save_path, checkpoint, scheduler)

    elif args.mode == 'evaluate':

        json_data = evaluate2(args.num_class, test_loader, model, model_parameter_path, class_name, path, device)
        # print(json_data, type(json_data))
        saveLocalFile(json_path + '/evaluate_{}'.format(args.name) + '.json', json_data)

    elif args.mode == 'semantic_features':
        if not os.path.exists(model_parameter_save_path + '/测试集语义特征.npy'):
            model_node_name = 'avgpool'
            encoding_array = semantic_features(device, model, model_node_name, test_img_paths, test_transform,
                                               model_parameter_path, model_parameter_save_path)
            # umap1(model_parameter_save_path, test_labels)  # 目前使用不了
        t_SNE1(model_parameter_save_path, test_labels, test_img_paths)

    # elif args.mode == 'visualization':
    #     img_index = 9
    #     if args.visual_method == 'CAM':
    #
    #         cam_visual(device, model, test_img_paths, test_labels, test_transform, class_name, model_parameter_path,
    #                    path, img_index=img_index, method_name='SmoothGradCAMpp')
    #
    #     elif args.visual_method == 'Grad_CAM':
    #         # 需要添加模型结构
    #
    #         grad_cam_visual(device, args.name, model, test_img_paths, test_labels, test_transform,
    #                         model_parameter_path, path, img_index=img_index, method_name='GradCAMPlusPlus')

    #     elif args.visual_method == 'DFF':
    #         # 使用cpu，并需要添加模型结构
    #         dff_visual(device, args.name, model, test_img_paths, model_parameter_path, path, img_index=img_index)
    #
    #     elif args.visual_method == 'Lime':
    #
    #         lime_visual(device, class_name_dict, model, test_img_paths, model_parameter_path, path, args.img_size,
    #                     args.input_size, img_index=img_index)
    #
    #     elif args.visual_method == 'Shap':
    #         # 不能保存到本地
    #         shap_visual(device, model, test_img_paths, test_labels, class_name, model_parameter_path, path,
    #                     img_index=img_index)
    #
    #     elif args.visual_method == 'Captum_Integrated_Gradients':
    #         # 使用cpu
    #         integrated_gradients_visual(device, model, test_img_paths, class_name_dict, model_parameter_path, path,
    #                                     args.img_size, args.input_size, img_index=img_index)
    #
    #     elif args.visual_method == 'Captum_GradientShap':
    #         # 使用cpu
    #         gradientShap_visual(device, model, test_img_paths, class_name_dict, model_parameter_path, path,
    #                             args.img_size, args.input_size, img_index=img_index)
    #
    #     else:
    #         raise Exception('Please input correct method!')
    #
    # elif args.mode == 'layer_visualization':
    #     img_index = 44
    #     if args.visual_method == 'CAM':
    #         for i in range(1, 16):
    #             cam_visual_layer(device, model, test_img_paths, test_labels, test_transform, class_name,
    #                              model_parameter_path,
    #                              path, img_index=img_index, method_name='SmoothGradCAMpp', layer=i)
    #
    #     elif args.visual_method == 'Grad_CAM':
    #         # 需要添加模型结构
    #         for i in range(1, 16):
    #             grad_cam_visual_layer(device, args.name, model, test_img_paths, test_labels, test_transform,
    #                                   model_parameter_path,
    #                                   path, img_index=img_index, method_name='GradCAMPlusPlus', layer=i)
    # else:
    #     raise Exception('Please input correct cmd! ')


if __name__ == '__main__':
    main()
