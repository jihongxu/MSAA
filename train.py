import pandas as pd
import time
# import wandb
import random
import matplotlib

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from V2.model_utils.utils import AverageMeter
from metrics import *

matplotlib.rc("font", family='SimHei')  # 中文字体


# wandb.init(project='graduation_project', name=time.strftime('%m-%d-%H-%M-%S'))


def train(train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    socres = AverageMeter()
    model.train()
    for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        score = quadratic_weighted_kappa(outputs, labels)

        losses.update(loss.item(), inputs.size(0))
        socres.update(score, inputs.size(0))
        return losses, socres


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    socres = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            score = quadratic_weighted_kappa(outputs, labels)

            losses.update(loss.item(), inputs.size(0))
            socres.update(score, inputs.size(0))
        return losses, socres


def Train(name, train_loader, val_loader, model, criterion, epochs, scheduler):
    log = pd.DataFrame(index=[], columns=[
        'epoch', 'loss', 'score', 'val_loss', 'val_score'
    ])
    log = {
        'epoch': [],
        'loss': [],
        'score': [],
        'val_loss': [],
        'val_score': [],
    }
    best_loss = float('inf')
    best_score = 0
    for epoch in range(epochs):
        print('Epoch [%d/%d]' % (epoch + 1, epochs))

        train_loss, train_score = train(train_loader, model, criterion)
        val_loss, val_score = validate(val_loader, model, criterion)
        if scheduler == 'CosineAnnealingLR':
            scheduler.step()
        elif scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_loss)

        print('loss %.4f - score %.4f - val_loss %.4f - val_score %.4f'
              % (train_loss, train_score, val_loss, val_score))

        log['epoch'].append(epoch)
        log['loss'].append(train_loss)
        log['score'].append(train_score)
        log['val_loss'].append(val_loss)
        log['val_score'].append(val_score)

        pd.DataFrame(log).to_csv('models/%s/log.csv' % (name), index=False)

        if val_loss < best_loss:
            torch.save(model.state_dict(), 'V2/outputs/models/%s_model.pth' % (name))
            best_loss = val_loss
            best_score = val_score
            print("=> saved best model")


def train2(name, train_loader, val_loader, model, criterion, optimizer, epochs, device):
    model.to(device)
    best_acc = 0

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        print('\nEpoch: %d' % (epoch + 1))
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_loader, 0):
            length = len(train_loader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            sum_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

        epoch_train_loss = sum_loss / len(train_loader)
        epoch_train_acc = 100. * correct / total
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        print('Waiting Test...')
        model.eval()
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            sum_loss = 0.0
            num = 0
            for i, data in enumerate(val_loader, 0):
                num += 1
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss = criterion(outputs, labels)
                sum_loss += test_loss.item()
            final_loss = round(sum_loss / num, 2)
            print('Validation loss: %.03f' % final_loss)

        epoch_val_loss = sum_loss / len(val_loader)
        epoch_val_acc = 100. * correct / total
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        now = round((100. * correct / total), 2)

        if now > best_acc:
            path = 'D:\ProgrammeWorkSpace\PycharmProjects\graduationProject\V2\outputs\models\model_%s' % name
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model.state_dict(), r'D:\ProgrammeWorkSpace\PycharmProjects\graduationProject\V2\outputs'
                                           r'\models\model_%s\train_%.3f.pth' % (name, now))
            print("Acc=: ", now)
            best_acc = now

    print('Train finished...')
    # 绘制训练损失和验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制训练准确度和验证准确度曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc.cpu(), label='Train Acc')
    plt.plot(val_acc.cpu(), label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def train3(name, train_loader, val_loader, model, criterion, optimizer, epochs, device, scheduler_name, model_save_path,
           scheduler=None):
    model.to(device)
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0.0

    for epoch in range(epochs):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()  # 训练模式
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_loader, 0):
            length = len(train_loader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            sum_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

        train_loss.append(sum_loss / length)
        train_acc.append(100. * correct / total)

        print('Waiting Test...')
        model.eval()  # 验证模式
        with torch.no_grad():
            correct = 0
            total = 0
            sum_loss = 0.0
            num = 0
            for i, data in enumerate(val_loader, 0):
                num += 1
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss = criterion(outputs, labels)
                sum_loss += test_loss.item()
            final_loss = round(sum_loss / num, 2)
            print('Validation loss: %.03f' % final_loss)

        now = round((100. * correct / total), 2)
        val_loss.append(sum_loss / num)
        val_acc.append(now)

        if scheduler_name == 'CosineAnnealingLR':
            scheduler.step()
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(sum_loss / num)

        if now > best_acc:

            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.state_dict(), model_save_path + '/' + 'train_%.3f.pth' % now)
            print("Acc=: ", now)
            best_acc = now

    print('Train finished...')

    # 绘制训练损失和验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss', linewidth=2, color='blue')
    plt.plot(val_loss, label='Val Loss', linewidth=2, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 1.5)
    plt.legend()
    plt.grid()
    plt.title('Training and Validation Loss')
    plt.show()

    train_acc = [x / 100 for x in train_acc]
    val_acc = [x / 100 for x in val_acc]
    train_acc = torch.tensor(train_acc)
    val_acc = torch.tensor(val_acc)
    # 绘制训练准确度和验证准确度曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc.cpu(), label='Train Acc', linewidth=2, color='blue')
    plt.plot(val_acc.cpu(), label='Val Acc', linewidth=2, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.title('Training and Validation Accuracy')
    plt.show()


def train4(train_loader, val_loader, model, criterion, optimizer, epochs, device, scheduler_name, model_save_path,
           scheduler=None):
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model.to(device)
    best_acc = 0.0

    df_train_log = pd.DataFrame()
    df_test_log = pd.DataFrame()
    length = 0
    for epoch in range(epochs):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()  # 训练模式
        sum_loss = 0.0
        correct = 0.0
        total = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs, labels = inputs.cpu(), labels.cpu()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 优化更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            loss = loss.detach().cpu().numpy()
            sum_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum()

            predicted = predicted.cpu().numpy()
            labels = labels.detach().cpu().numpy()
            train_acc = accuracy_score(labels, predicted)

            print('[epoch:%d, batch:%d] Loss: %.03f | Acc: %.3f%%  train_acc: %.3f'
                  % (epoch + 1, length, loss, 100. * correct / total, train_acc))

            log_train = {}
            log_train['epoch'] = epoch
            log_train['batch'] = length
            log_train['train_loss'] = loss
            log_train['train_accuracy'] = train_acc
            log_train['train_ACC'] = float(100. * correct / total)
            # wandb.log(log_train)
            df_train_log = df_train_log.append(log_train, ignore_index=True)
            length = length + 1
        print('Waiting Test...')
        model.eval()  # 验证模式
        with torch.no_grad():

            loss_list = []
            labels_list = []
            preds_list = []
            for i, data in enumerate(val_loader, 0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                loss = loss.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                predicted = predicted.cpu().numpy()

                loss_list.append(loss)
                labels_list.extend(labels)
                preds_list.extend(predicted)

        log_test = {}
        log_test['epoch'] = epoch
        log_test['test_loss'] = np.mean(loss_list)
        log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
        log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
        log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
        log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
        # wandb.log(log_test)
        df_test_log = df_test_log.append(log_test, ignore_index=True)

        print('Validation loss: %.04f ' % log_test['test_loss'])
        print('Validation acc: %.04f ' % log_test['test_accuracy'])

        if scheduler_name == 'CosineAnnealingLR':
            scheduler.step()
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(log_test['test_loss'])

        if log_test['test_accuracy'] > best_acc:
            # 删除旧的最佳模型文件(如有)
            old_best_acc_path = model_save_path + '/best_{:.3f}.pth'.format(best_acc)
            if os.path.exists(old_best_acc_path):
                os.remove(old_best_acc_path)

            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.state_dict(), model_save_path + '/best_{:.3f}.pth'.format(log_test['test_accuracy']))
            print('保存新的最佳模型', 'best-{:.3f}.pth'.format(log_test['test_accuracy']))
            best_acc = log_test['test_accuracy']

        PATH = model_save_path + '/state'
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        old_PATH = PATH + '/epoch_{}.pth'.format(epoch - 1)
        PATH = PATH + '/epoch_{}.pth'.format(epoch)

        if os.path.exists(old_PATH):
            os.remove(old_PATH)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'scheduler': scheduler,
        }, PATH)

    print('Train finished...')
    # print(df_train_log)
    # print(df_test_log)
    df_train_log.to_csv(model_save_path + '/训练日志-训练集.csv', index=False)
    df_test_log.to_csv(model_save_path + '/训练日志-测试集.csv', index=False)
    visualization_train(model_save_path, model_save_path + '/训练日志-训练集.csv', model_save_path + '/训练日志-测试集.csv')


def train5(train_loader, val_loader, test_loader, model, criterion, optimizer, epochs, device, scheduler_name,
           model_save_path,
           scheduler=None):
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model.to(device)
    best_acc = 0.0
    best_test_acc = 0.0

    df_train_log = pd.DataFrame()
    df_val_log = pd.DataFrame()
    df_test_log = pd.DataFrame()
    length = 0
    for epoch in range(epochs):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()  # 训练模式
        sum_loss = 0.0
        correct = 0.0
        total = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs, labels = inputs.cpu(), labels.cpu()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 优化更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            loss = loss.detach().cpu().numpy()
            sum_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum()

            predicted = predicted.cpu().numpy()
            labels = labels.detach().cpu().numpy()
            train_acc = accuracy_score(labels, predicted)

            print('[epoch:%d, batch:%d] Loss: %.06f | Acc: %.3f%%  train_acc: %.3f'
                  % (epoch + 1, length, loss, 100. * correct / total, train_acc))

            log_train = {}
            log_train['epoch'] = epoch
            log_train['batch'] = length
            log_train['train_loss'] = loss
            log_train['train_accuracy'] = train_acc
            log_train['train_ACC'] = float(100. * correct / total)
            # wandb.log(log_train)
            df_train_log = df_train_log.append(log_train, ignore_index=True)
            length = length + 1

        print('Waiting Val...')
        model.eval()  # 验证模式
        with torch.no_grad():

            loss_list = []
            labels_list = []
            preds_list = []
            for i, data in enumerate(val_loader, 0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                loss = loss.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                predicted = predicted.cpu().numpy()

                loss_list.append(loss)
                labels_list.extend(labels)
                preds_list.extend(predicted)

        log_val = {}
        log_val['epoch'] = epoch
        log_val['test_loss'] = np.mean(loss_list)
        log_val['test_accuracy'] = accuracy_score(labels_list, preds_list)
        log_val['test_precision'] = precision_score(labels_list, preds_list, average='macro')
        log_val['test_recall'] = recall_score(labels_list, preds_list, average='macro')
        log_val['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
        # wandb.log(log_test)
        df_val_log = df_val_log.append(log_val, ignore_index=True)

        print('Validation loss: %.06f ' % log_val['test_loss'])
        print('Validation acc: %.04f ' % log_val['test_accuracy'])
        if scheduler_name == 'CosineAnnealingLR':
            scheduler.step()
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(log_val['test_loss'])

        print('Waiting Testing...')
        model.eval()  # 验证模式
        with torch.no_grad():

            labels_list = []
            preds_list = []
            for i, data in enumerate(test_loader, 0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                labels = labels.detach().cpu().numpy()
                predicted = predicted.cpu().numpy()

                labels_list.extend(labels)
                preds_list.extend(predicted)

        log_test = {}
        log_test['epoch'] = epoch
        log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
        log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
        log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
        log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
        # wandb.log(log_test)
        df_test_log = df_test_log.append(log_test, ignore_index=True)
        print('Test acc: %.4f ' % log_test['test_accuracy'])

        if log_val['test_accuracy'] > best_acc:
            # 删除旧的最佳模型文件(如有)
            old_best_acc_path = model_save_path + '/best_{:.4f}.pth'.format(best_acc)
            if os.path.exists(old_best_acc_path):
                os.remove(old_best_acc_path)

            torch.save(model.state_dict(), model_save_path + '/best_{:.4f}.pth'.format(log_val['test_accuracy']))
            print('保存新的Val最佳模型', 'best-{:.4f}.pth'.format(log_val['test_accuracy']))
            best_acc = log_val['test_accuracy']

        if log_test['test_accuracy'] > best_test_acc:
            # 删除旧的最佳模型文件(如有)
            old_best_test_acc_path = model_save_path + '/bestTest_{:.4f}.pth'.format(best_test_acc)
            if os.path.exists(old_best_test_acc_path):
                os.remove(old_best_test_acc_path)

            torch.save(model.state_dict(), model_save_path + '/bestTest_{:.4f}.pth'.format(log_test['test_accuracy']))
            print('保存新的Test最佳模型', 'best-{:.3f}.pth'.format(log_test['test_accuracy']))
            best_test_acc = log_test['test_accuracy']

        PATH = model_save_path + '/state'
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        old_PATH = PATH + '/epoch_{}.pth'.format(epoch - 1)
        PATH = PATH + '/epoch_{}.pth'.format(epoch)

        if os.path.exists(old_PATH):
            os.remove(old_PATH)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'scheduler': scheduler,
        }, PATH)

    print('Train finished...')
    # print(df_train_log)
    # print(df_test_log)
    df_train_log.to_csv(model_save_path + '/训练日志-训练集.csv', index=False)
    df_val_log.to_csv(model_save_path + '/训练日志-验证集.csv', index=False)
    df_test_log.to_csv(model_save_path + '/训练日志-测试集.csv', index=False)
    visualization_train(model_save_path, model_save_path + '/训练日志-训练集.csv', model_save_path + '/训练日志-验证集.csv')


def continue_train(train_loader, val_loader, model, criterion, optimizer, epochs, device, scheduler_name,
                   model_save_path, checkpoint, scheduler=None):
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器状态
    # scheduler.load_state_dict(checkpoint['scheduler'])  # 加载调度程序状态
    epoch = checkpoint['epoch']  # 加载已经训练的epoch
    loss = checkpoint['loss']  # 加载已经训练的损失
    model.to(device)

    best_acc = 0.0
    df_train_log = pd.read_csv(model_save_path + '/训练日志-训练集.csv')
    df_test_log = pd.read_csv(model_save_path + '/训练日志-测试集.csv')
    length = int(df_train_log.iloc[-1].at['batch']) + 1
    for epoch in range(epoch, epochs):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()  # 训练模式
        sum_loss = 0.0
        correct = 0.0
        total = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs, labels = inputs.cpu(), labels.cpu()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 优化更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            loss = loss.detach().cpu().numpy()
            sum_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum()

            predicted = predicted.cpu().numpy()
            labels = labels.detach().cpu().numpy()
            train_acc = accuracy_score(labels, predicted)

            print('[epoch:%d, batch:%d] Loss: %.03f | Acc: %.3f%%  train_acc: %.3f'
                  % (epoch + 1, length, loss, 100. * correct / total, train_acc))

            log_train = {}
            log_train['epoch'] = epoch
            log_train['batch'] = length
            log_train['train_loss'] = loss
            log_train['train_accuracy'] = train_acc
            log_train['train_ACC'] = float(100. * correct / total)
            # wandb.log(log_train)
            df_train_log = df_train_log.append(log_train, ignore_index=True)
            length = length + 1
        print('Waiting Test...')
        model.eval()  # 验证模式
        with torch.no_grad():

            loss_list = []
            labels_list = []
            preds_list = []
            for i, data in enumerate(val_loader, 0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                loss = loss.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                predicted = predicted.cpu().numpy()

                loss_list.append(loss)
                labels_list.extend(labels)
                preds_list.extend(predicted)

        log_test = {}
        log_test['epoch'] = epoch
        log_test['test_loss'] = np.mean(loss_list)
        log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
        log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
        log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
        log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
        # wandb.log(log_test)
        df_test_log = df_test_log.append(log_test, ignore_index=True)

        print('Validation loss: %.04f ' % log_test['test_loss'])
        print('Validation acc: %.04f ' % log_test['test_accuracy'])

        if scheduler_name == 'CosineAnnealingLR':
            scheduler.step()
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(log_test['test_loss'])

        if log_test['test_accuracy'] > best_acc:
            # 删除旧的最佳模型文件(如有)
            old_best_acc_path = model_save_path + '/best_{:.3f}.pth'.format(best_acc)
            if os.path.exists(old_best_acc_path):
                os.remove(old_best_acc_path)

            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.state_dict(), model_save_path + '/best_{:.3f}.pth'.format(log_test['test_accuracy']))
            print('保存新的最佳模型', 'best-{:.3f}.pth'.format(log_test['test_accuracy']))
            best_acc = log_test['test_accuracy']

        PATH = model_save_path + '/state'
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        old_PATH = PATH + '/epoch_{}.pth'.format(epoch - 1)
        PATH = PATH + '/epoch_{}.pth'.format(epoch)

        if os.path.exists(old_PATH):
            os.remove(old_PATH)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'scheduler': scheduler,
        }, PATH)

    print('Train finished...')
    # print(df_train_log)
    # print(df_test_log)
    df_train_log.to_csv(model_save_path + '/训练日志-训练集.csv', index=False)
    df_test_log.to_csv(model_save_path + '/训练日志-测试集.csv', index=False)
    visualization_train(model_save_path, model_save_path + '/训练日志-训练集.csv', model_save_path + '/训练日志-测试集.csv')


def continue_train1(train_loader, val_loader, test_loader, model, criterion, optimizer, epochs, device, scheduler_name,
                    model_save_path, checkpoint, scheduler=None):
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器状态
    # scheduler.load_state_dict(checkpoint['scheduler'])  # 加载调度程序状态
    epoch = checkpoint['epoch']  # 加载已经训练的epoch
    loss = checkpoint['loss']  # 加载已经训练的损失
    model.to(device)

    best_acc = 0.0
    best_test_acc = 0.0
    df_train_log = pd.read_csv(model_save_path + '/训练日志-训练集.csv')
    df_val_log = pd.read_csv(model_save_path + '/训练日志-验证集.csv')
    df_test_log = pd.read_csv(model_save_path + '/训练日志-测试集.csv')
    # df_test_log = pd.DataFrame()

    length = int(df_train_log.iloc[-1].at['batch']) + 1
    for epoch in range(epoch, epochs):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()  # 训练模式
        sum_loss = 0.0
        correct = 0.0
        total = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs, labels = inputs.cpu(), labels.cpu()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 优化更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            loss = loss.detach().cpu().numpy()
            sum_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum()

            predicted = predicted.cpu().numpy()
            labels = labels.detach().cpu().numpy()
            train_acc = accuracy_score(labels, predicted)

            print('[epoch:%d, batch:%d] Loss: %.6f | Acc: %.3f%%  train_acc: %.3f'
                  % (epoch + 1, length, loss, 100. * correct / total, train_acc))

            log_train = {}
            log_train['epoch'] = epoch
            log_train['batch'] = length
            log_train['train_loss'] = loss
            log_train['train_accuracy'] = train_acc
            log_train['train_ACC'] = float(100. * correct / total)
            # wandb.log(log_train)
            df_train_log = df_train_log.append(log_train, ignore_index=True)
            length = length + 1

        print('Waiting Val...')
        model.eval()  # 验证模式
        with torch.no_grad():

            loss_list = []
            labels_list = []
            preds_list = []
            for i, data in enumerate(val_loader, 0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                loss = loss.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                predicted = predicted.cpu().numpy()

                loss_list.append(loss)
                labels_list.extend(labels)
                preds_list.extend(predicted)

        log_val = {}
        log_val['epoch'] = epoch
        log_val['test_loss'] = np.mean(loss_list)
        log_val['test_accuracy'] = accuracy_score(labels_list, preds_list)
        log_val['test_precision'] = precision_score(labels_list, preds_list, average='macro')
        log_val['test_recall'] = recall_score(labels_list, preds_list, average='macro')
        log_val['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
        # wandb.log(log_test)
        df_val_log = df_val_log.append(log_val, ignore_index=True)

        print('Validation loss: %.06f ' % log_val['test_loss'])
        print('Validation acc: %.04f ' % log_val['test_accuracy'])

        if scheduler_name == 'CosineAnnealingLR':
            scheduler.step()
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(log_val['test_loss'])

        print('Waiting Testing...')
        model.eval()  # 验证模式
        with torch.no_grad():

            labels_list = []
            preds_list = []
            for i, data in enumerate(test_loader, 0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                labels = labels.detach().cpu().numpy()
                predicted = predicted.cpu().numpy()

                loss_list.append(loss)
                labels_list.extend(labels)
                preds_list.extend(predicted)

        log_test = {}
        log_test['epoch'] = epoch
        log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
        log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
        log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
        log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
        # wandb.log(log_test)
        df_test_log = df_test_log.append(log_test, ignore_index=True)
        print('Test acc: %.4f ' % log_test['test_accuracy'])

        if log_val['test_accuracy'] > best_acc:
            # 删除旧的最佳模型文件(如有)
            old_best_acc_path = model_save_path + '/best_{:.4f}.pth'.format(best_acc)
            if os.path.exists(old_best_acc_path):
                os.remove(old_best_acc_path)

            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.state_dict(), model_save_path + '/best_{:.4f}.pth'.format(log_val['test_accuracy']))
            print('保存新的Val最佳模型', 'best-{:.4f}.pth'.format(log_val['test_accuracy']))
            best_acc = log_val['test_accuracy']

        if log_test['test_accuracy'] > best_test_acc:
            # 删除旧的最佳模型文件(如有)
            old_best_test_acc_path = model_save_path + '/bestTest_{:.4f}.pth'.format(best_test_acc)
            if os.path.exists(old_best_test_acc_path):
                os.remove(old_best_test_acc_path)

            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.state_dict(), model_save_path + '/bestTest_{:.4f}.pth'.format(log_test['test_accuracy']))
            print('保存新的Test最佳模型', 'best-{:.4f}.pth'.format(log_test['test_accuracy']))
            best_test_acc = log_test['test_accuracy']

        PATH = model_save_path + '/state'
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        old_PATH = PATH + '/epoch_{}.pth'.format(epoch - 1)
        PATH = PATH + '/epoch_{}.pth'.format(epoch)

        if os.path.exists(old_PATH):
            os.remove(old_PATH)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'scheduler': scheduler,
        }, PATH)

    print('Train finished...')
    # print(df_train_log)
    # print(df_test_log)
    df_train_log.to_csv(model_save_path + '/训练日志-训练集.csv', index=False)
    df_val_log.to_csv(model_save_path + '/训练日志-验证集.csv', index=False)
    df_test_log.to_csv(model_save_path + '/训练日志-测试集.csv', index=False)
    visualization_train(model_save_path, model_save_path + '/训练日志-训练集.csv', model_save_path + '/训练日志-验证集.csv')


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


def visualization_train(model_save_path, train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 训练集损失函数
    plt.figure(figsize=(16, 8))
    x = df_train['batch']
    y = df_train['train_loss']

    plt.plot(x, y, label='训练集')

    plt.tick_params(labelsize=20)
    plt.xlabel('batch', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.title('训练集损失函数', fontsize=25)
    plt.savefig(model_save_path + '/训练集损失函数.pdf', dpi=120, bbox_inches='tight')
    plt.show()
    # 训练集准确率
    plt.figure(figsize=(16, 8))
    x = df_train['batch']
    y = df_train['train_accuracy']

    plt.plot(x, y, label='训练集')

    plt.tick_params(labelsize=20)
    plt.xlabel('batch', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.title('训练集准确率', fontsize=25)
    plt.savefig(model_save_path + '/训练集准确率.pdf', dpi=120, bbox_inches='tight')
    plt.show()

    # 测试集损失函数
    plt.figure(figsize=(16, 8))

    x = df_test['epoch']
    y = df_test['test_loss']

    plt.plot(x, y, label='测试集')

    plt.tick_params(labelsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.title('测试集损失函数', fontsize=25)
    plt.savefig(model_save_path + '/测试集损失函数.pdf', dpi=120, bbox_inches='tight')

    plt.show()

    # 测试集评估指标
    metrics_name = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1-score']

    plt.figure(figsize=(16, 8))

    x = df_test['epoch']
    for y in metrics_name:
        plt.plot(x, df_test[y], label=y, **get_line_arg())

    plt.tick_params(labelsize=20)
    plt.ylim([0, 1])
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel(y, fontsize=20)
    plt.title('测试集分类评估指标', fontsize=25)
    plt.savefig(model_save_path + '/测试集分类评估指标.pdf', dpi=120, bbox_inches='tight')

    plt.legend(fontsize=20)
    plt.show()
