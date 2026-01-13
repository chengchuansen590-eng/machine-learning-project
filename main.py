# -*- coding: utf-8 -*-
"""
Modified to load HDF5 format data (sub0-sub31) with shape [40,15,1,28,512]
"""

import copy
import os
import gc
import h5py
import shutil
from time import strftime
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    f1_score
import yaml
from easydict import EasyDict
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

import preprocess
from modellibs import models
from torchutils import get_trainer
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils import (print_time_stamp, reset_workpath, save_history, save_log_confusion_matrix, seed_everything,
                   save_metrics)


def save_results(fold_idx, true_labels, pred_labels, history, cfg):
    """Save evaluation results and metrics"""
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, normalize='true')
    np.save(os.path.join(cfg.ckpt_dir, f'confusion_matrix_fold{fold_idx}.npy'), cm)

    # Metrics
    metrics = {
        'acc': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels, average='macro'),
        'recall': recall_score(true_labels, pred_labels, average='macro'),
        'f1': f1_score(true_labels, pred_labels, average='macro')
    }
    pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(cfg.ckpt_dir, f'metrics_fold{fold_idx}.csv'), index=False)

    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(cfg.ckpt_dir, f'history_fold{fold_idx}.csv'), index=False)


# %% Modified Data Loading Functions
def load_hdf5_data(subject_ids, data_dir):
    """Load HDF5 format data for specified subjects"""
    all_data = []
    all_labels = []

    for sub_id in tqdm(subject_ids, desc="Loading HDF5 data"):
        with h5py.File(os.path.join(data_dir, f'sub{sub_id}.hdf'), 'r') as f:
            # Data shape: [40 trials, 15 segments, 1, 28 channels, 512 timepoints]
            data = f['data'][:]
            label = f['label'][:]  # Shape: [40 trials, 15 segments] or [40,]

            # Reshape data to [n_samples, 1, 28, 512]
            data = data.reshape(-1, 28, 512)

            # Handle different label formats
            if label.ndim == 2:  # [40,15]
                label = label.reshape(-1)
            elif label.ndim == 1:  # [40,]
                label = np.repeat(label, 15)  # Repeat label for each segment

            all_data.append(data)
            all_labels.append(label)

    return np.concatenate(all_data), np.concatenate(all_labels)


def dataSecondLoad(sub_train, dataType):
    """Modified to load from HDF5 files"""
    print(f"Loading {dataType} data for subjects: {sub_train['sub'].tolist()}")

    # Get subject IDs (assuming sub_train DataFrame has 'sub' column with 0-31)
    subject_ids = sub_train['sub'].astype(int).tolist()

    # Load data and labels
    data, labels = load_hdf5_data(subject_ids, DATA_DIR)

    # Shuffle data
    permutation = np.random.permutation(data.shape[0])
    data = data[permutation]
    labels = labels[permutation]

    # Balance classes (keep same number of samples per class)
    num_classes = len(np.unique(labels))
    num_sample_logs = [np.sum(labels == _) for _ in range(num_classes)]
    min_samples = np.min(num_sample_logs)

    balanced_data = []
    balanced_labels = []
    for label_id in range(num_classes):
        idxs = np.where(labels == label_id)[0][:min_samples]
        balanced_data.append(data[idxs])
        balanced_labels.append(labels[idxs])

    return np.concatenate(balanced_data), np.concatenate(balanced_labels)


# %% Main Execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACTN')
    parser.add_argument('--workpath', '-W', type=str, default='workpath')
    parser.add_argument('--reload', '-R', action='store_true')
    parser.add_argument('--resume-K', '-K', type=int, default=0 , help='Resume from fold K')
    parser.add_argument('--data-dir', type=str, default='data_raw_DEAP_A',
                        help='Directory containing HDF5 files')
    args = parser.parse_args()

    # Load configuration
    with open('workpath/config.yaml') as f:
        CFG = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # Set data directory
    DATA_DIR = args.data_dir

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CFG.device) if isinstance(CFG.device, int) else ','.join(
        str(_) for _ in CFG.device)
    torch.cuda.empty_cache()

    # Initialize
    seed_everything(CFG.seed)

    # Model parameters
    model_name = CFG.get('model_name', 'default_model')
    input_shape = CFG.get('input_shape', [28,512])  # Updated for HDF5 data shape
    output_shape = CFG.get('num_classes', 9)
    trainer_name = CFG.get('trainer_name', 'CrossVal')

    # Directory setup
    CFG.ckpt_dir = f'{args.workpath}/ckpt'
    CFG.model_save_dir = f'{args.workpath}/model'
    os.makedirs(CFG.ckpt_dir, exist_ok=True)
    os.makedirs(CFG.model_save_dir, exist_ok=True)
    model_metrics_dir = os.path.join(args.workpath, 'ckpt')
    if not os.path.exists(model_metrics_dir):
        os.makedirs(model_metrics_dir)

    # Prepare subject list (0-31)
    subject_ids = list(range(32))  # sub0 to sub31
    sub_df = pd.DataFrame({'sub': subject_ids, 'num': [40 * 15] * 32}) # 40 trials × 15 segments
    sub_per = sub_df.sample(frac=1, random_state=CFG.seed, ignore_index=True)
    # 创建0-98的受试者ID列表
    # subject_ids = list(range(99))  # sub0到sub98
    #
    # # 初始化num列表
    # num_values = []
    #
    # # 遍历每个受试者
    # for sub_id in subject_ids:
    #     # 构建文件名
    #     filename = f'data_raw_DEAP_A\sub{sub_id}.hdf'
    #
    #     try:
    #         # 读取HDF5文件
    #         with h5py.File(filename, 'r') as f:
    #             # 获取data数据集
    #             data = f['data']
    #             # 计算前两个维度的乘积
    #             product = data.shape[0] * data.shape[1]
    #             num_values.append(product)
    #     except Exception as e:
    #         print(f"Error processing {filename}: {str(e)}")
    #         num_values.append(0)  # 如果文件读取失败，设为0
    #
    # # 创建DataFrame
    # sub_df = pd.DataFrame({'sub': subject_ids, 'num': num_values})

    # K-fold cross validation
    start_kfold = args.resume_K if 0 <= args.resume_K < len(subject_ids) else 0
    for i in range(start_kfold,len(subject_ids)):
        # # Split subjects into train/valid/test
        # test_start = i * int(32 / CFG.n_fold)
        # test_end = (i + 1) * int(32 / CFG.n_fold)
        # test_subs = list(range(test_start, test_end))
        #
        # remaining_subs = [s for s in subject_ids if s not in test_subs]
        # valid_subs = random.sample(remaining_subs, int(len(remaining_subs) * 0.2))
        # train_subs = [s for s in remaining_subs if s not in valid_subs]
        #
        # # Create DataFrames for each set
        # sub_test = sub_df[sub_df['sub'].isin(test_subs)].reset_index(drop=True)
        # sub_valid = sub_df[sub_df['sub'].isin(valid_subs)].reset_index(drop=True)
        # sub_train = sub_df[sub_df['sub'].isin(train_subs)].reset_index(drop=True)
        valid_subs = [subject_ids[i]]
        # 当前训练集的受试者
        train_subs = [s for s in subject_ids if s != subject_ids[i]]

        # 创建子集 DataFrame
        sub_valid = sub_df[sub_df['sub'].isin(valid_subs)].reset_index(drop=True)
        sub_train = sub_df[sub_df['sub'].isin(train_subs)].reset_index(drop=True)

        # 加载数据
        data_train, label_train = dataSecondLoad(sub_train, 'TRAIN')
        data_valid, label_valid = dataSecondLoad(sub_valid, 'VALID')

        # Setup training
        CFG.ckpt_name = f'ckpt_{model_name}_{i}'
        writer = SummaryWriter(log_dir=os.path.join(CFG.ckpt_dir, f'event/fold_{i}'))

        # Initialize and train model
        print_time_stamp(f'Training model: {model_name}, fold: {i}')
        model = models.get_model(model_name, input_shape, output_shape)
        trainer = get_trainer(trainer_name, data_train, label_train, data_valid, label_valid, model, CFG, writer)
        trainer.fit()
        history = trainer.history

        # # Evaluate on test set
        # data_test, label_test = dataSecondLoad(sub_test, 'TEST')
        # label_pred = trainer.predict(data_test)
        gc.collect()
        trainer.load_ckpt()
        label_pred = trainer.predict(data_valid)
        label_pred = [1 if x > 1 else x for x in label_pred]
        del data_valid
        gc.collect()

        # Save results and metrics
        # save_results(i, label_test, label_pred, trainer.history, CFG)
        # draw figures
        cm_raw = confusion_matrix(
            label_valid, label_pred, normalize='true')
        cm = np.around(cm_raw * 100, decimals=1)
        # labels = ['anger', 'disgust', 'fear', 'sadness', 'neutral',
        #           'amusement', 'joy', 'inspiration', 'tenderness']
        labels = ['Negative', 'Postive']
        cmdp = ConfusionMatrixDisplay(cm, display_labels=labels)
        plt.rcParams['figure.figsize'] = [10, 10]
        cmdp.plot(cmap=plt.cm.Reds, xticks_rotation=75,
                  colorbar=True, values_format='.1f')
        plt.savefig(os.path.join(model_metrics_dir,
                                 f'ConfusionMat_fold{i}.pdf'))
        plt.close()
        acc = history['acc']
        val_acc = history['val_acc']
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Train_Acc')
        plt.plot(epochs, val_acc, 'b', label='Val_Acc')
        plt.title('Train and Val Acc', fontsize=20)
        plt.legend()
        plt.savefig(os.path.join(model_metrics_dir, f'Acc_fold{i}.pdf'))
        plt.close()
        plt.plot(epochs, loss, 'bo', label='Train_Loss')
        plt.plot(epochs, val_loss, 'b', label='Val_Loss')
        plt.title('Train and Val Loss', fontsize=20)
        plt.legend()
        plt.savefig(os.path.join(model_metrics_dir, f'Loss_fold{i}.pdf'))
        plt.close()

        writer.close()

        try:
            his_name = os.path.join(model_metrics_dir, 'history.csv')
            his_copy = copy.deepcopy(history)
            his_rename = {k + f'_{i}': v for k, v in his_copy.items()}
            his_df = pd.DataFrame(his_rename)
            save_history(his_df, his_name)
            cm_name = os.path.join(model_metrics_dir, 'confusion_matrix.npy')
            cm_copy = copy.deepcopy(cm)
            save_log_confusion_matrix(cm, cm_name)

            acc = accuracy_score(label_valid, label_pred)
            precision = precision_score(
                label_valid, label_pred, average='macro')
            recall = recall_score(label_valid, label_pred, average='macro')
            f1 = f1_score(label_valid, label_pred, average='macro')
            metrics = {'acc': [acc], 'precision': [
                precision], 'recall': [recall], 'f1': [f1]}
            metric_df = pd.DataFrame(copy.deepcopy(metrics))
            metric_name = os.path.join(model_metrics_dir, 'metrics.csv')
            save_metrics(metric_df, metric_name)
        except:
            print('save history failure')

        if CFG.debug:
            break

    try:
        cm_data_name = os.path.join(model_metrics_dir, 'confusion_matrix.npy')
        cm_data = np.load(cm_data_name)
        cm_save_name = os.path.join(
            model_metrics_dir, f'ConfusionMat_{CFG.n_fold}_folds.pdf')
        labels = ['Negative', 'Postive']
        cm_data = cm_data.mean(axis=0)
        cmdp = ConfusionMatrixDisplay(cm_data, display_labels=labels)
        plt.rcParams['figure.figsize'] = [10, 10]
        cmdp.plot(cmap=plt.cm.Reds, xticks_rotation=75,
                  colorbar=True, values_format='.1f')
        plt.savefig(cm_save_name)
        plt.close()
    except:
        print('Draw confusion matrix all fold failure')
        # if CFG.debug:
        #     break


def save_results(fold_idx, true_labels, pred_labels, history, cfg):
    model_metrics_dir = 'D:\Anaconda\MACTN-main\workpath\ckpt'
    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels, normalize='true')
    np.save(os.path.join(model_metrics_dir, f'confusion_matrix_fold{fold_idx}.npy'), cm)
    labels = ['anger', 'disgust', 'fear', 'sadness', 'neutral', 'amusement', 'joy', 'inspiration', 'tenderness']
    plt.figure(figsize=(10, 10))
    cmdp = ConfusionMatrixDisplay(cm, display_labels=labels)
    cmdp.plot(cmap=plt.cm.Reds, xticks_rotation=75, colorbar=True, values_format='.1f')
    plt.savefig(os.path.join(model_metrics_dir, f'ConfusionMat_fold{fold_idx}.pdf'))
    plt.close()

    # Training Metrics
    metrics = {
        'acc': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels, average='macro'),
        'recall': recall_score(true_labels, pred_labels, average='macro'),
        'f1': f1_score(true_labels, pred_labels, average='macro')
    }
    pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(model_metrics_dir, f'metrics_fold{fold_idx}.csv'), index=False)

    # Training History
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(model_metrics_dir, f'history_fold{fold_idx}.csv'), index=False)

    # Plot Accuracy and Loss
    epochs = range(1, len(history['acc']) + 1)

    plt.plot(epochs, history['acc'], 'bo', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'b', label='Val Acc')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_metrics_dir, f'Acc_fold{fold_idx}.pdf'))
    plt.close()

    plt.plot(epochs, history['loss'], 'bo', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'b', label='Val Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(model_metrics_dir, f'Loss_fold{fold_idx}.pdf'))
    plt.close()

    # """Save evaluation results and metrics"""
    # # Confusion matrix
    # cm = confusion_matrix(true_labels, pred_labels, normalize='true')
    # np.save(os.path.join(cfg.ckpt_dir, f'confusion_matrix_fold{fold_idx}.npy'), cm)
    #
    # # Metrics
    # metrics = {
    #     'acc': accuracy_score(true_labels, pred_labels),
    #     'precision': precision_score(true_labels, pred_labels, average='macro'),
    #     'recall': recall_score(true_labels, pred_labels, average='macro'),
    #     'f1': f1_score(true_labels, pred_labels, average='macro')
    # }
    # pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(cfg.ckpt_dir, f'metrics_fold{fold_idx}.csv'), index=False)
    #
    # # Save training history
    # pd.DataFrame(history).to_csv(os.path.join(cfg.ckpt_dir, f'history_fold{fold_idx}.csv'), index=False)
