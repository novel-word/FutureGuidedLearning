from utils import MackeyGlass, RNN, create_time_series_dataset, KL, plot_predictions
from torch.utils.data import Subset, DataLoader, TensorDataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# Check for GPU availability
device = torch.device('cpu')

# Hyperparameters
def FGL(student_horizon, alpha, num_bins):
    lookback_window = 1
    teacher_horizon = 1
    input_size = lookback_window  # one value 
    hidden_size = 128  # hidden neurons in RNN
    output_size = num_bins  # output neurons = num_bins
    num_layers = 3  # RNN layers
    batch_size = 1  # must be 1 for continual learning!!!

    learning_rate = 0.0001

    num_epochs = 10

    with open('data.pkl', 'rb') as f:
        mackey_glass_data = pickle.load(f)

    train, test, original, discretized = create_time_series_dataset(
        data=mackey_glass_data,
        lookback_window=lookback_window,
        forecasting_horizon=teacher_horizon,
        num_bins=num_bins,
        test_size=0.25)
    
    teacher = RNN(input_size, hidden_size, output_size, num_layers).to(device)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=learning_rate)

    mse = torch.nn.MSELoss()
    celoss = torch.nn.CrossEntropyLoss()

    #Teacher = torch.load(f'teacher_{num_bins}.pt', weights_only=False)
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train:
            inputs = inputs.float().to(device)
            inputs = inputs.reshape(1, 1, lookback_window)
            targets = targets.long().to(device)
            outputs = teacher(inputs)
            loss = celoss(outputs, targets[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train):.4f}')

    teacher.eval()
    with torch.no_grad():
        total_loss = 0
        Tpredictions = []
        Ttrue_values = []
        for inputs, targets in test:
            inputs = inputs.float().to(device)
            inputs = inputs.reshape(1, 1, lookback_window)
            targets = targets.float().to(device)
            outputs = teacher(inputs)
            probabilities = torch.nn.functional.softmax(outputs.detach(), dim=1)
            _, predicted_class = probabilities.max(1)
            loss = mse(predicted_class.float(), targets[0])
            total_loss += loss.item()
            Tpredictions.append(predicted_class.cpu().numpy())
            Ttrue_values.append(targets[0].cpu().detach().numpy())
        Tmse = total_loss / len(test)
        print(f'Model Test Loss: {Tmse:.4f}')
    
    ### Training the Student ###
    train, test, _, _ = create_time_series_dataset(
        data=mackey_glass_data,
        lookback_window=lookback_window,
        forecasting_horizon=student_horizon,
        num_bins=num_bins,
        test_size=0.25)

    baseline = RNN(input_size, hidden_size, output_size, num_layers).to(device)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=learning_rate)
    kl = torch.nn.KLDivLoss(reduction='batchmean')

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train:
            inputs = inputs.float().to(device)
            inputs = inputs.reshape(1, 1, lookback_window)
            targets = targets.long().to(device)
            outputs = baseline(inputs)
            loss = celoss(outputs, targets[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Print loss and learning rate
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train):.4f}')

    baseline.eval()
    with torch.no_grad():
        total_loss = 0
        Bpredictions = []
        Btrue_values = []
        for inputs, targets in test:
            inputs = inputs.float().to(device)
            inputs = inputs.reshape(1, 1, lookback_window)
            targets = targets.float().to(device)
            outputs = baseline(inputs)
            probabilities = torch.nn.functional.softmax(outputs.detach(), dim=1)
            _, predicted_class = probabilities.max(1)
            loss = mse(predicted_class.float(), targets[0])
            total_loss += loss.item()
            Bpredictions.append(predicted_class.cpu().numpy())
            Btrue_values.append(targets[0].cpu().detach().numpy())
        Bmse = total_loss / len(test)
        print(f'Model Test Loss: {Bmse:.4f}')

    # Train the Student with Teacher guidance
    train_h, test_h, _, _ = create_time_series_dataset(
        data=mackey_glass_data,
        lookback_window=lookback_window,
        forecasting_horizon=teacher_horizon,
        num_bins=num_bins,
        test_size=0.25,
        offset=student_horizon-1)

    Student = RNN(input_size, hidden_size, output_size, num_layers).to(device)
    optimizer = torch.optim.Adam(Student.parameters(), lr=learning_rate)
    celoss = torch.nn.CrossEntropyLoss()
    teacher.eval()

    for epoch in range(num_epochs):
        total_loss = 0
        for (inputs, targets), (inputs_h, targets_h) in zip(train, train_h):
            inputs = inputs.float().to(device)
            inputs = inputs.reshape(1, 1, lookback_window)
            inputs_h = inputs_h.float().to(device)
            inputs_h = inputs_h.reshape(1, 1, lookback_window)
            targets = targets.long().to(device)
            outputs = Student(inputs)
            with torch.no_grad():
                logits = teacher(inputs_h)
            loss = alpha * celoss(outputs, targets[0]) + KL(outputs, logits, alpha=alpha, kl=kl, T=4)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Print loss and learning rate
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train):.4f}')

    Student.eval()
    with torch.no_grad():
        total_loss = 0
        Spredictions = []
        Strue_values = []
        for inputs, targets in test:
            inputs = inputs.float().to(device)
            inputs = inputs.reshape(1, 1, lookback_window)
            targets = targets.float().to(device)
            outputs = Student(inputs)
            probabilities = torch.nn.functional.softmax(outputs.detach(), dim=1)
            _, predicted_class = probabilities.max(1)
            loss = mse(predicted_class.float(), targets[0])
            total_loss += loss.item()
            Spredictions.append(predicted_class.cpu().numpy())
            Strue_values.append(targets[0].cpu().detach().numpy())
        Smse = total_loss / len(test)
        print(f'Model Test Loss: {Smse:.4f}')
    print(f'{Bmse:.2f}, {Smse:.2f}')
    #plot_predictions(Tpredictions, Ttrue_values, Bpredictions, Btrue_values, Spredictions, Strue_values, original, discretized)
    return [Bmse, Smse]

horizons = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
results = dict()
for i in horizons:
    results[i] = FGL(i, 0.0, 50)
print(results)
