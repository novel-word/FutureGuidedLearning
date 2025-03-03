import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import MackeyGlass, RNN, create_time_series_dataset, KL, plot_predictions
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_student_model(student_horizon, alpha, num_bins, epochs, optimizer_type, temperature):
    """
    Trains and evaluates the teacher-student framework for time series forecasting.

    Args:
        student_horizon (int): Forecasting horizon for the student model.
        alpha (float): Weight for cross-entropy loss.
        num_bins (int): Number of discrete bins in the output.
        epochs (int): Number of training epochs.
        optimizer_type (str): Optimizer type ('SGD' or 'Adam').
        temperature (float): Temperature for knowledge distillation.

    Returns:
        list: Baseline MSE and Student MSE.
    """
    print(f"\nTraining Student Model | Horizon: {student_horizon} | Alpha: {alpha} | Epochs: {epochs} | Optimizer: {optimizer_type} | Temperature: {temperature}")
    
    lookback_window = 1
    teacher_horizon = 1
    input_size = lookback_window  # One value
    hidden_size = 128  # Hidden neurons in RNN
    output_size = num_bins  # Output neurons = num_bins
    num_layers = 3  # RNN layers
    batch_size = 1  # Must be 1 for continual learning!!!
    learning_rate = 0.0001

    # Load data
    with open("data.pkl", "rb") as f:
        mackey_glass_data = pickle.load(f)

    # Train and test dataset creation
    train, test, original, discretized = create_time_series_dataset(
        data=mackey_glass_data,
        lookback_window=lookback_window,
        forecasting_horizon=teacher_horizon,
        num_bins=num_bins,
        test_size=0.25
    )

    # Initialize teacher model
    teacher = RNN(input_size, hidden_size, output_size, num_layers).to(device)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=learning_rate)

    mse = nn.MSELoss()
    celoss = nn.CrossEntropyLoss()

    # Train Teacher Model
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train:
            inputs = inputs.float().to(device).reshape(1, 1, lookback_window)
            targets = targets.long().to(device)
            outputs = teacher(inputs)
            loss = celoss(outputs, targets[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train):.4f}")

    teacher.eval()
    
    # Train and evaluate Baseline Model
    train, test, _, _ = create_time_series_dataset(
        data=mackey_glass_data,
        lookback_window=lookback_window,
        forecasting_horizon=student_horizon,
        num_bins=num_bins,
        test_size=0.25
    )

    baseline = RNN(input_size, hidden_size, output_size, num_layers).to(device)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train:
            inputs = inputs.float().to(device).reshape(1, 1, lookback_window)
            targets = targets.long().to(device)
            outputs = baseline(inputs)
            loss = celoss(outputs, targets[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train):.4f}")

    baseline.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in test:
            inputs = inputs.float().to(device).reshape(1, 1, lookback_window)
            targets = targets.float().to(device)
            outputs = baseline(inputs)
            probabilities = F.softmax(outputs.detach(), dim=1)
            _, predicted_class = probabilities.max(1)
            loss = mse(predicted_class.float(), targets[0])
            total_loss += loss.item()
        Bmse = total_loss / len(test)
        print(f"Baseline Model Test Loss: {Bmse:.4f}")

    # Train Student Model with Teacher Guidance
    train_h, test_h, _, _ = create_time_series_dataset(
        data=mackey_glass_data,
        lookback_window=lookback_window,
        forecasting_horizon=teacher_horizon,
        num_bins=num_bins,
        test_size=0.25,
        offset=student_horizon - 1
    )

    student = RNN(input_size, hidden_size, output_size, num_layers).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate) if optimizer_type == "Adam" else torch.optim.SGD(student.parameters(), lr=learning_rate, momentum=0.9)
    
    teacher.eval()

    for epoch in range(epochs):
        total_loss = 0
        for (inputs, targets), (inputs_h, targets_h) in zip(train, train_h):
            inputs = inputs.float().to(device).reshape(1, 1, lookback_window)
            inputs_h = inputs_h.float().to(device).reshape(1, 1, lookback_window)
            targets = targets.long().to(device)
            outputs = student(inputs)
            
            with torch.no_grad():
                logits = teacher(inputs_h)
            
            loss = alpha * celoss(outputs, targets[0]) + KL(outputs, logits, alpha=alpha, kl=nn.KLDivLoss(reduction="batchmean"), T=temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train):.4f}")

    student.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in test:
            inputs = inputs.float().to(device).reshape(1, 1, lookback_window)
            targets = targets.float().to(device)
            outputs = student(inputs)
            probabilities = F.softmax(outputs.detach(), dim=1)
            _, predicted_class = probabilities.max(1)
            loss = mse(predicted_class.float(), targets[0])
            total_loss += loss.item()
        Smse = total_loss / len(test)
        print(f"Student Model Test Loss: {Smse:.4f}")

    print(f"Baseline MSE: {Bmse:.2f}, Student MSE: {Smse:.2f}")
    return [Bmse, Smse]


if __name__ == "__main__":
    """
    Main execution loop that takes command-line arguments for:
    - Forecasting horizon
    - Alpha value for loss weighting
    - Number of discrete bins
    - Number of training epochs
    - Optimizer type
    - Temperature for knowledge distillation
    """

    parser = argparse.ArgumentParser(description="Teacher-Student Framework for Time Series Forecasting")
    parser.add_argument("--horizon", type=int, required=True, help="Forecasting horizon for the student model")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value for loss weighting")
    parser.add_argument("--num_bins", type=int, default=50, help="Number of discrete bins (default: 50)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam"], default="Adam", help="Optimizer type (default: Adam)")
    parser.add_argument("--temperature", type=float, default=4, help="Temperature for distillation (default: 4)")

    args = parser.parse_args()

    results = train_student_model(args.horizon, args.alpha, args.num_bins, args.epochs, args.optimizer, args.temperature)

    # Save results
    with open("forecasting_results.txt", "a") as f:
        f.write(f"Horizon {args.horizon} | Alpha {args.alpha} | Results: {results}\n")
