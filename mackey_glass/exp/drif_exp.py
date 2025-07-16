import argparse
from collections import deque
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from utils import create_time_series_dataset, RNN


def page_hinkley_update(error, state, delta=0.005):
    state['t'] += 1
    m_prev = state['m']
    state['m'] += (error - state['m']) / state['t']
    state['PH'] = max(0.0, state['PH'] + (error - m_prev - delta))
    return state


def run_ph_baseline(
    student_horizon, num_bins, lookback_window, batch_size,
    window_size, delta, lambda_thr, retrain_epochs, device,
    verbose=False
):
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    train_loader, test_loader, _, _ = create_time_series_dataset(
        data=data,
        lookback_window=lookback_window,
        forecasting_horizon=student_horizon,
        num_bins=num_bins,
        test_size=0.25,
        batch_size=batch_size
    )

    # Initial training for 10 epochs
    model = RNN(lookback_window, 128, num_bins, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    celoss = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(10):
        for _, inputs, targets in train_loader:
            x = inputs.float().to(device).view(-1,1,lookback_window)
            y = targets.long().to(device).squeeze(-1)
            out = model(x)
            loss = celoss(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()

    # Set up drift detection
    state = {'m': 0.0, 'PH': 0.0, 't': 0}
    window = deque(maxlen=window_size)
    mse_loss = nn.MSELoss()
    errors = []

    for batch_idx, (_, inputs, targets) in enumerate(test_loader):
        x = inputs.float().to(device).view(-1,1,lookback_window)
        y = targets.long().to(device).squeeze(-1)

        with torch.no_grad():
            logits = model(x)
        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        err = mse_loss(preds.float(), y.float())
        errors.append(err.item())

        state = page_hinkley_update(err.item(), state, delta)
        window.append((x.cpu(), y.cpu()))

        if state['PH'] > lambda_thr and len(window) == window_size:
            if verbose:
                print(f"Drift detected on batch {batch_idx:03d}")
            with torch.enable_grad():
                model.train()
                for _ in range(retrain_epochs):
                    for win_x, win_y in window:
                        win_x, win_y = win_x.to(device), win_y.to(device)
                        out = model(win_x)
                        loss = celoss(out, win_y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            model.eval()
            state = {'m': 0.0, 'PH': 0.0, 't': 0}
            window.clear()

    avg_mse = sum(errors) / len(errors)
    print(f"Page–Hinkley Horizon {student_horizon} MSE: {avg_mse:.4f}")
    return avg_mse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("PH Drift Baseline for Mackey–Glass")
    parser.add_argument("--horizon",        type=int,   default=6)
    parser.add_argument("--bins",           type=int,   default=50)
    parser.add_argument("--lookback_window",type=int,   default=50)
    parser.add_argument("--batch_size",     type=int,   default=128)
    parser.add_argument("--window",         type=int,   default=50,
                        help="Reduced window to trigger faster retraining")
    parser.add_argument("--delta",          type=float, default=0.001,
                        help="Smaller delta → more sensitive")
    parser.add_argument("--lambda_thr",     type=float, default=0.05,
                        help="Lower threshold to actually trigger")
    parser.add_argument("--retrain_epochs", type=int,   default=3)
    parser.add_argument("--verbose",        action="store_true",
                        help="Print error & PH per batch")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_ph_baseline(
        student_horizon=args.horizon,
        num_bins=args.bins,
        lookback_window=args.lookback_window,
        batch_size=args.batch_size,
        window_size=args.window,
        delta=args.delta,
        lambda_thr=args.lambda_thr,
        retrain_epochs=args.retrain_epochs,
        device=device,
        verbose=args.verbose
    )
    
