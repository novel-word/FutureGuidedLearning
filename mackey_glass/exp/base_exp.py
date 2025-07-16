import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import MackeyGlass, RNN, create_time_series_dataset, plot_predictions, KL

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_state = None

    def step(self, current_loss, model):
        # improvement?
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss  = current_loss
            self.counter    = 0
            self.best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            return False    # don’t stop
        else:
            self.counter += 1
            return (self.counter >= self.patience)

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)

def train_student_model(student_horizon, alpha, num_bins, epochs,
                        temperature, lookback_window, batch_size,
                        patience=5):
    torch.manual_seed(42)
    print(f"\nTraining | Horizon={student_horizon} Alpha={alpha} "
          f"Epochs={epochs} Temp={temperature} "
          f"Lookback={lookback_window} Batch={batch_size}")

    # hyperparams
    hidden_size   = 128
    output_size   = num_bins
    num_layers    = 2
    lr            = 1e-4

    # load data
    with open("data.pkl","rb") as f:
        data = pickle.load(f)

    # teacher (1-step, offset=H-1)
    teacher_train, teacher_test, _, _ = create_time_series_dataset(
        data, lookback_window, 1,
        num_bins, 0.25, offset=student_horizon-1, batch_size=batch_size
    )
    # student (H-step, offset=0)
    student_train, student_test, _, _ = create_time_series_dataset(
        data, lookback_window, student_horizon,
        num_bins, 0.25, offset=0, batch_size=batch_size
    )

    mse   = nn.MSELoss()
    celoss= nn.CrossEntropyLoss()

    # ---- Teacher loop with early stopping ----
    teacher     = RNN(lookback_window, hidden_size, output_size, num_layers).to(device)
    opt_t       = torch.optim.Adam(teacher.parameters(), lr=lr)
    stop_t      = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        teacher.train()
        for _, x, y in teacher_train:
            x = x.float().to(device).view(-1,1,lookback_window)
            y = y.long().to(device).squeeze(-1)
            out = teacher(x)
            loss= celoss(out, y)
            opt_t.zero_grad(); loss.backward(); opt_t.step()

        # validate
        teacher.eval()
        with torch.no_grad():
            val_loss = 0.
            for _, x, y in teacher_test:
                x = x.float().to(device).view(-1,1,lookback_window)
                y = y.float().to(device).squeeze(-1)
                pred = teacher(x).argmax(dim=1).float()
                val_loss += mse(pred, y).item()
            val_loss /= len(teacher_test)

        #print(f"[Teacher] Epoch {epoch+1}: Val MSE={val_loss:.4f}")
        if stop_t.step(val_loss, teacher):
            print(f"[Teacher] Early stopping at epoch {epoch+1}")
            break
    stop_t.restore(teacher)
    print(f"[Teacher] Best Val MSE = {stop_t.best_loss:.4f}")

    # ---- Baseline loop with early stopping ----
    baseline   = RNN(lookback_window, hidden_size, output_size, num_layers).to(device)
    opt_b      = torch.optim.Adam(baseline.parameters(), lr=lr)
    stop_b     = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        baseline.train()
        for _, x, y in student_train:
            x = x.float().to(device).view(-1,1,lookback_window)
            y = y.long().to(device).squeeze(-1)
            out = baseline(x)
            loss= celoss(out, y)
            opt_b.zero_grad(); loss.backward(); opt_b.step()

        # validate
        baseline.eval()
        with torch.no_grad():
            val_loss = 0.
            for _, x, y in student_test:
                x = x.float().to(device).view(-1,1,lookback_window)
                y = y.float().to(device).squeeze(-1)
                pred = baseline(x).argmax(dim=1).float()
                val_loss += mse(pred, y).item()
            val_loss /= len(student_test)

        #print(f"[Baseline] Epoch {epoch+1}: Val MSE={val_loss:.4f}")
        if stop_b.step(val_loss, baseline):
            print(f"[Baseline] Early stopping at epoch {epoch+1}")
            break
    stop_b.restore(baseline)
    print(f"[Baseline] Best Val MSE = {stop_b.best_loss:.4f}")

    # ---- Student + distillation loop w/ early stopping ----
    student = RNN(lookback_window, hidden_size, output_size, num_layers).to(device)
    opt_s    = torch.optim.Adam(student.parameters(), lr=lr)
    stop_s   = EarlyStopper(patience=patience)

    for epoch in range(epochs):
        student.train()
        for (i_s, x_s, y_s), (i_t, x_t, y_t) in zip(student_train, teacher_train):
            x_s = x_s.float().to(device).view(-1,1,lookback_window)
            targets = y_s.long().to(device).squeeze(-1)
            outputs = student(x_s)

            x_t = x_t.float().to(device).view(-1,1,lookback_window)
            with torch.no_grad():
                logits = teacher(x_t)

            loss = alpha * celoss(outputs, targets) + KL(outputs, logits, temperature, alpha)
            opt_s.zero_grad(); loss.backward(); opt_s.step()

        # validate student
        student.eval()
        with torch.no_grad():
            val_loss = 0.
            for _, x, y in student_test:
                x    = x.float().to(device).view(-1,1,lookback_window)
                pred = student(x).argmax(dim=1).float()
                val_loss += mse(pred, y.float().to(device).squeeze(-1)).item()
            val_loss /= len(student_test)

        #print(f"[Student] Epoch {epoch+1}: Val MSE={val_loss:.4f}")
        if stop_s.step(val_loss, student):
            print(f"[Student] Early stopping at epoch {epoch+1}")
            break
    stop_s.restore(student)
    print(f"[Student] Best Val MSE = {stop_s.best_loss:.4f}")

    # final test metrics
    def evaluate(model, loader):
        model.eval()
        tot = 0.
        with torch.no_grad():
            for _, x, y in loader:
                x = x.float().to(device).view(-1,1,lookback_window)
                tot += mse(model(x).argmax(dim=1).float(), y.float().to(device).squeeze(-1)).item()
        return tot / len(loader)

    Tmse = evaluate(teacher, teacher_test); print(f"Teacher Test MSE:  {Tmse:.4f}")
    Bmse = evaluate(baseline, student_test); print(f"Baseline Test MSE: {Bmse:.4f}")
    Smse = evaluate(student, student_test); print(f"Student Test MSE:  {Smse:.4f}")

    return Bmse, Smse

# … rest of your __main__ unchanged …


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Future-Guided Learning for Time-Series Forecasting"
    )
    parser.add_argument("--horizon", type=int, required=True, help="Student horizon (N)")
    parser.add_argument("--alpha",   type=float, required=True, help="Loss weight α")
    parser.add_argument("--num_bins",type=int, default=50,   help="Number of bins")
    parser.add_argument("--epochs",  type=int, default=10,   help="Training epochs")
    parser.add_argument("--temperature", type=float, default=4)
    parser.add_argument("--lookback_window", type=int, default=1,
                        help="Length of history fed to RNN")
    parser.add_argument("--batch_size",      type=int, default=1,
                        help="Batch size for all loaders")
    args = parser.parse_args()

    results = train_student_model(
        student_horizon   = args.horizon,
        alpha             = args.alpha,
        num_bins          = args.num_bins,
        epochs            = args.epochs,
        temperature       = args.temperature,
        lookback_window   = args.lookback_window,
        batch_size        = args.batch_size
    )

    with open("forecasting_results.txt", "a") as f:
        f.write(f"Horizon={args.horizon}  Alpha={args.alpha}  Results={results}\n")
