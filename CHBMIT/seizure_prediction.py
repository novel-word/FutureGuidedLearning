import json
import numpy as np
from utils.load_signals_student import PrepDataStudent
from utils.prep_data_student import train_val_test_split_continual_s, train_test_split_s
from models.model import CNN_LSTM_Model
from models.MViT import MViT
from os import makedirs
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import os 
def find_best_threshold(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def main(target, trials, model='CNN_LSTM'):
    print ('Main - Seizure Prediction')
    torch.cuda.empty_cache()
   
    epochs=25
    with open('student_settings.json') as k:
        student_settings = json.load(k)

    student_results = []
    student_losses = []

    ictal_X, ictal_y = PrepDataStudent(target, type='ictal', settings=student_settings).apply()
    interictal_X, interictal_y = PrepDataStudent(target, type='interictal', settings=student_settings).apply()
    X_train, y_train, X_test, y_test = train_val_test_split_continual_s(ictal_X, ictal_y, interictal_X, interictal_y, 0.35)
    
    
    Y_train       = torch.tensor(y_train).type(torch.LongTensor).to('cuda')
    X_train       = torch.tensor(X_train).type(torch.FloatTensor).to('cuda')
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader  = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    for i in range(trials):
        if model == 'MViT':
            student = MViT(X_shape=X_train.shape, in_channels=X_train.shape[2], num_classes=2, patch_size=(5,10), 
                 embed_dim=128, num_heads=4, hidden_dim=256, num_layers=4, dropout=0.1).to('cuda')
        else: 
            student = CNN_LSTM_Model(X_train.shape).to('cuda')
        ce_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(student.parameters(), lr=5e-4, betas=(0.9,0.999), eps=1e-8)
        #optimizer = torch.optim.SGD(student.parameters(), lr=5e-4, momentum=0.9)
        pbar = tqdm(total=epochs)
        for epoch in range(epochs):
            student.train()
            #total_loss = 0
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")
                student_logits = student(X_batch)
                loss = ce_loss(student_logits, Y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #total_loss += loss.cpu().item()
            pbar.update(1)
            #student_losses.append(total_loss / len(train_loader))
            #print(f'\nEpoch: {epoch} | Loss {total_loss/len(train_loader):.2f}')
        pbar.close()   
        student.eval()
        student.to('cuda')
        X_tensor = torch.tensor(X_test).float().to('cuda')
        y_tensor = torch.tensor(y_test).long().to('cuda')
        with torch.no_grad():
            predictions = student(X_tensor)
        predictions = F.softmax(predictions, dim=1)
        predictions = predictions[:, 1].cpu().numpy()
        threshold = find_best_threshold(y_tensor.cpu(), predictions)
        binary_predictions = (predictions >= threshold).astype(int)
        cm = confusion_matrix(y_tensor.cpu(), binary_predictions)
        tn, fp, fn, tp = cm.ravel()
        # Calculate FPR, Sensitivity, AUCROC
        fpr = fp / (fp + tn)
        sensitivity = tp / (tp + fn)
        auc_roc = roc_auc_score(y_tensor.cpu(), predictions)
        print(f'Patient {target}')
        print(f'False Positive Rate (FPR): {fpr:.2f}')
        print(f'Sensitivity: {sensitivity:.2f}')
        print(f'AUCROC: {auc_roc:.2f}')
        student_results.append([fpr, sensitivity, auc_roc])
    return student_results
    
if __name__ == "__main__":
    targets = ['1', '2', '3', '5', '9','10','13','18','19','20','21','23']
    results = {}
    for target in targets:
        results[target] = main(target, trials=3, model='MViT')
    with open('Baseline_results.txt', 'a') as f:
        f.write(f'{str(results)}')
        f.close()



