#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import gc
import os
import sys
import time
from collections import Counter
# sys.path.append("../../")
# sys.path.insert(0, "../logdeep/")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from logdeep.dataset.log import log_dataset
from logdeep.dataset.sample import session_window
from logdeep.tools.utils import (save_parameters, seed_everything,
                                 train_val_split)

global result_filename
result_filename = f"times_result_{time.time()}.csv"

def generate(name):
    window_size = 10
    hdfs = {}
    length = 0
    with open('data/hdfs/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
            length += 1
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs, length


class Predicter():
    def __init__(self, model, options):
        self.data_dir = options['data_dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.batch_size = options['batch_size']
        self.save_dir = options['save_dir']

    def predict_unsupervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_normal_loader, test_normal_length = generate('hdfs_test_normal')
        test_abnormal_loader, test_abnormal_length = generate(
            'hdfs_test_abnormal')
        df = pd.DataFrame(columns=["start_time",	"logs_seq", 	"new_log",	"dataset_type","end_time"])
        df.to_csv(rf"{self.save_dir}/{result_filename}")
        TP = 0
        FP = 0
        dict_list = []
        # Test the model
        start_time = time.time()
        with torch.no_grad():
            for k, line in enumerate(test_normal_loader.keys()):
                for p in range(test_normal_loader[line]):
                    for i in range(len(line) - self.window_size):
                        seq0 = line[i:i + self.window_size]
                        label = line[i + self.window_size]
                        seq1 = [0] * 28
                        log_conuter = Counter(seq0)
                        for key in log_conuter:
                            seq1[key] = log_conuter[key]

                        seq0 = torch.tensor(seq0, dtype=torch.float).view(
                            -1, self.window_size, self.input_size).to(self.device)
                        seq1 = torch.tensor(seq1, dtype=torch.float).view(
                            -1, self.num_classes, self.input_size).to(self.device)
                        label = torch.tensor(label).view(-1).to(self.device)
                        df_row_dict = {"start_time": time.time(),
                                    "logs_seq": seq0.ravel().tolist(),
                                    "new_log": label.ravel().tolist(),
                                    "dataset_type":"normal"}
                        output = model(features=[seq0, seq1], device=self.device)
                        df_row_dict["end_time"] = time.time()
                        dict_list.append(df_row_dict)
                        predicted = torch.argsort(output,
                                                1)[0][-self.num_candidates:]
                        # df = pd.concat(df, pd.Datadf_row_dict, ignore_index=True)
                        if len(dict_list)%50000 == 0:
                            print("dumping")
                            dict_list = self.dump_csv(dict_list)
                if label not in predicted:
                    FP += test_normal_loader[line]
                    break
        dict_list = self.dump_csv(dict_list)
        with torch.no_grad():
            for line in test_abnormal_loader.keys():
                for p in range(test_abnormal_loader[line]):
                    for i in range(len(line) - self.window_size):
                        seq0 = line[i:i + self.window_size]
                        label = line[i + self.window_size]
                        seq1 = [0] * 28
                        log_conuter = Counter(seq0)
                        for key in log_conuter:
                            seq1[key] = log_conuter[key]

                        seq0 = torch.tensor(seq0, dtype=torch.float).view(
                            -1, self.window_size, self.input_size).to(self.device)
                        seq1 = torch.tensor(seq1, dtype=torch.float).view(
                            -1, self.num_classes, self.input_size).to(self.device)
                        label = torch.tensor(label).view(-1).to(self.device)
                        df_row_dict = {"start_time": time.time(),
                                    "logs_seq": seq0.ravel().tolist(),
                                    "new_log": label.ravel().tolist(),
                                    "dataset_type":"abnormal"}
                        output = model(features=[seq0, seq1], device=self.device)
                        df_row_dict["end_time"] = time.time()
                        dict_list.append(df_row_dict)
                        predicted = torch.argsort(output,
                                                1)[0][-self.num_candidates:]
                if label not in predicted:
                    TP += test_abnormal_loader[line]
                    break
        dict_list = self.dump_csv(dict_list)
        # Compute precision, recall and F1-measure
        FN = test_abnormal_length - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

    def dump_csv(self, dict_list):
        df = pd.DataFrame.from_records(dict_list)
        dict_list = []
        df.to_csv(rf"{self.save_dir}/{result_filename}", mode='a', header=False)
        df = df[0:0]
        return dict_list

    def predict_supervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_logs, test_labels = session_window(self.data_dir, datatype='test')
        test_dataset = log_dataset(logs=test_logs,
                                   labels=test_labels,
                                   seq=self.sequentials,
                                   quan=self.quantitatives,
                                   sem=self.semantics)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=True)
        tbar = tqdm(self.test_loader, desc="\r")
        TP, FP, FN, TN = 0, 0, 0, 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().to(self.device))
            output = self.model(features=features, device=self.device)
            output = F.sigmoid(output)[:, 0].cpu().detach().numpy()
            # predicted = torch.argmax(output, dim=1).cpu().numpy()
            predicted = (output < 0.2).astype(int)
            label = np.array([y.cpu() for y in label])
            TP += ((predicted == 1) * (label == 1)).sum()
            FP += ((predicted == 1) * (label == 0)).sum()
            FN += ((predicted == 0) * (label == 1)).sum()
            TN += ((predicted == 0) * (label == 0)).sum()
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
