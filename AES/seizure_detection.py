import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample
import stft
from utils.log import log
from utils.save_load import save_pickle_file, load_pickle_file, \
    save_hickle_file, load_hickle_file
from utils.prep_data_teacher import train_val_test_split_continual_t
import json
import random
import os
import os.path
from utils.load_signals_student import PrepDataStudent
from utils.load_signals_teacher import PrepDataTeacher, makedirs
from utils.prep_data_student import train_val_test_split_continual_s

from models.models import CNN_LSTM_Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import optuna 

#create the universal teacher based on 
def main(target):
    torch.cuda.empty_cache()
    epochs = 25
    with open('teacher_settings.json') as f:
        teacher_settings = json.load(f)
    makedirs(str(teacher_settings['cachedir']))

    teacher_results = []
    freq = 1000
    teacher_channels=15
    #ictal_X, ictal_y, interictal_X, interictal_y = make_teacher(mode='Dog', teacher_settings=teacher_settings)
    ictal_X, ictal_y = PrepDataTeacher(target, type='ictal', settings=teacher_settings, freq=freq, teacher_channels=teacher_channels).apply()
    interictal_X, interictal_y = PrepDataTeacher(target, type='interictal', settings=teacher_settings, freq=freq, teacher_channels=teacher_channels).apply()
    X_train, y_train = train_val_test_split_continual_t(ictal_X, ictal_y, interictal_X, interictal_y, 0, no_test=True)   
    
    teacher = CNN_LSTM_Model(X_train.shape).to('cuda')

    Y_train       = torch.tensor(y_train).type(torch.LongTensor).to('cuda')
    X_train       = torch.tensor(X_train).type(torch.FloatTensor).to('cuda')
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader_teacher  = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-4, betas=(0.9,0.999), eps=1e-8)
    #optimizer = torch.optim.Adam(teacher.parameters(), lr=5e-4)
    pbar = tqdm(total=epochs)
    for epoch in range(epochs):
        teacher.train()
        total_loss = 0
        for X_batch, Y_batch in train_loader_teacher:
            X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")
            outputs = teacher(X_batch)
            loss = ce_loss(outputs, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Loss for epoch {epoch}: {total_loss/len(train_loader_teacher)}')
        pbar.update(1)

    pbar.close()   
    return teacher_results

if __name__ == '__main__':
    #4 dogs and 4 humans
    targets = ['Dog_1', 'Dog_2','Dog_3','Dog_4','Patient_3','Patient_5','Patient_6','Patient_7']
    for target in targets:
        main(target)

