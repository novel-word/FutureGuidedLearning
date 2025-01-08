import json
import numpy as np
from utils.load_signals_student import PrepDataStudent
from utils.prep_data_student import train_val_test_split_continual_s
from models.model import CNN_LSTM_Model
from os import makedirs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

#performs knowledge distillation on each patient for every ratio of alpha
def main(target):
    print ('Main - Seizure Prediction (FGL)')
    torch.cuda.empty_cache()

    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    epochs=25
    T = 4 #temperature
    trials = 3
    result_dict = dict()

    with open('student_settings.json') as k:
        student_settings = json.load(k)

    path = f'pytorch_models/Patient_{target}_detection'
    teacher = torch.load(path)
    ictal_X, ictal_y = PrepDataStudent(target, type='ictal', settings=student_settings).apply()
    interictal_X, interictal_y = PrepDataStudent(target, type='interictal', settings=student_settings).apply()
    X_train, y_train, X_test, y_test = train_val_test_split_continual_s(ictal_X, ictal_y, interictal_X, interictal_y, 0.35)
    Y_train       = torch.tensor(y_train).type(torch.LongTensor).to('cuda')
    X_train       = torch.tensor(X_train).type(torch.FloatTensor).to('cuda')
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader  = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

    for i in ratios:
        alpha, beta = i, 1-i
        auc_list = []
        for j in range(trials):
            student = CNN_LSTM_Model(X_train.shape).to('cuda')

            ce_loss = nn.CrossEntropyLoss()
            #optimizer = torch.optim.Adam(student.parameters(), lr=5e-4, betas=(0.9,0.999), eps=1e-8)
            optimizer = torch.optim.SGD(student.parameters(), lr=5e-4, momentum=0.9)
            pbar = tqdm(total=epochs)
            for epoch in range(epochs):
                student.train()
                teacher.eval()
                total_loss = 0
                for X_batch, Y_batch in train_loader:
                    X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")
                    student_logits = student(X_batch)
                    with torch.no_grad():
                        teacher_logits = teacher(X_batch)
                    soft_targets = F.softmax(teacher_logits / T, dim=1)
                    soft_prob = F.log_softmax(student_logits / T, dim=1)
                    distillation_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T**2) 
                    
                    loss = alpha*ce_loss(student_logits, Y_batch) + beta*distillation_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                pbar.update(1)
            pbar.close()   

            student.eval()
            student.to('cuda')
            X_tensor = torch.tensor(X_test).float().to('cuda')
            y_tensor = torch.tensor(y_test).long().to('cuda')

            with torch.no_grad():
                predictions = student(X_tensor)
            predictions = F.softmax(predictions, dim=1)
            predictions = predictions[:, 1].cpu().numpy()
            auc_test = roc_auc_score(y_tensor.cpu(), predictions)
            print('Test AUC is:', auc_test)
            auc_list.append(auc_test)
        result_dict[i] = auc_list
    print(result_dict)
    with open("FGL_results.txt", 'a') as f:  
        f.write(f'Patient_{target}_Results=')
        f.write(f'{str(result_dict)}')
        f.write('\n')
        f.close()

if __name__ == "__main__":
    targets = ['1','2','3','5','9','10','13','18','19','20','21','23']
    for target in targets:
        main(target)



