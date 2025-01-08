import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample
import stft
from utils.log import log
from utils.save_load import save_pickle_file, load_pickle_file, save_hickle_file, load_hickle_file
from models.models import CNN_LSTM_Model, EarlyStopping, CNN_Model
from utils.prep_data_teacher import train_val_test_split_continual_t
from utils.load_signals_teacher import PrepDataTeacher, make_teacher, makedirs
import json
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import optuna 
import matplotlib.pyplot as plt
def main(target):

    epochs = 50

    torch.cuda.empty_cache()
    
    with open('teacher_settings.json') as f:
        teacher_settings = json.load(f)
    with open('student_settings.json') as k:
        student_settings = json.load(k)

    makedirs(str(teacher_settings['cachedir']))
    makedirs(str(student_settings['cachedir']))

    #Train the Teacher 
    if 'Dog' in target:
        mode = 'Dog'
        path = f'teacher_dog.pth'
    elif target == 'Patient_1':
        mode = 'Patient_1'
        path = f'{target}.pth'
    else:
        mode = 'Patient_2'
        path = f'{target}.pth'

    #preprocessing
    teacher_loss = []
    ictal_X, ictal_y, interictal_X, interictal_y = make_teacher(mode=mode, teacher_settings=teacher_settings)
    X_train, y_train  = train_val_test_split_continual_t(ictal_X, ictal_y, interictal_X, interictal_y, 0.0, no_test=True)   
    Y_train       = torch.tensor(y_train).type(torch.LongTensor).to('cuda')
    X_train       = torch.tensor(X_train).type(torch.FloatTensor).to('cuda')
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader_teacher  = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

    #model creation
    teacher = CNN_LSTM_Model(X_train.shape).to('cuda')
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(teacher.parameters(), lr=5e-4, momentum=0.9)
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
        pbar.update(1)
        print(f'Loss for epoch {epoch}:', total_loss/len(train_loader_teacher))
    pbar.close()   
    torch.save(teacher, path)
    #plt.plot(teacher_loss, label='teacher loss')
    #plt.show()

if __name__ == '__main__':
    targets = ['Dog', 'Patient_1', 'Patient_2']
    for target in targets:
        main(target)