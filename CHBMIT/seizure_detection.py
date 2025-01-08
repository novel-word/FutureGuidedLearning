import json
import numpy as np
from utils.model_stuff import EarlyStopping
from utils.load_signals_teacher import PrepDataTeacher
from utils.prep_data_teacher import train_val_test_split_continual_t, train_test_split_t
from models.model import CNN_LSTM_Model
from os import makedirs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from models.MViT import MViT
def main():
    print ('Main - Seizure Detection')
    torch.cuda.empty_cache()

    epochs=50
    
    with open('teacher_settings.json') as k:
        teacher_settings = json.load(k)

    teacher_results = []
    targets = ['1','2','3','5','9','10','13','18','19','20','21','23']
    
    for target in targets:
        early_stopping = EarlyStopping(patience=5, mode='min')
        teacher_losses = []
        ictal_X, ictal_y = PrepDataTeacher(target, type='ictal', settings=teacher_settings).apply()
        interictal_X, interictal_y = PrepDataTeacher(target, type='interictal', settings=teacher_settings).apply()
        X_train, y_train, X_test, y_test = train_val_test_split_continual_t(ictal_X, ictal_y, interictal_X, interictal_y, 0.35)
        teacher = CNN_LSTM_Model(X_train.shape)

        Y_train       = torch.tensor(y_train).type(torch.LongTensor).to('cuda')
        X_train       = torch.tensor(X_train).type(torch.FloatTensor).to('cuda')
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader  = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

        ce_loss = nn.CrossEntropyLoss()
        #optimizer = torch.optim.Adam(teacher.parameters(), lr=5e-4, betas=(0.9,0.999), eps=1e-8)
        optimizer = torch.optim.SGD(teacher.parameters(), lr=5e-4, momentum=0.9)
        pbar = tqdm(total=epochs)
        for epoch in range(epochs):
            teacher.train()
            total_loss = 0
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")
                teacher_logits = teacher(X_batch)
                loss = ce_loss(teacher_logits, Y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            pbar.update(1)
            teacher_losses.append(total_loss / len(train_loader))
            early_stopping.step(total_loss / len(train_loader), epoch)
            if early_stopping.is_stopped():
                print(f"Early stopping at epoch {epoch} with best loss {early_stopping.best_loss:.4f}")
                break
        pbar.close()   

        teacher.eval()
        teacher.to('cuda')
        X_tensor = torch.tensor(X_test).float().to('cuda')
        y_tensor = torch.tensor(y_test).long().to('cuda')

        with torch.no_grad():
            predictions = teacher(X_tensor)
        predictions = F.softmax(predictions, dim=1)
        predictions = predictions[:, 1].cpu().numpy()
        auc_test = roc_auc_score(y_tensor.cpu(), predictions)
        print('Test AUC is:', auc_test)
        teacher_results.append(auc_test)
        path = f'pytorch_models/Patient_{target}_detection'
        torch.save(teacher, path)
        plt.plot(teacher_losses, label='detector loss')
        #plt.show()
if __name__ == "__main__":
    main()

