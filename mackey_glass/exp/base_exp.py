import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
# from utils import MackeyGlass, RNN, create_time_series_dataset, plot_predictions, KL
from utils.utils import MackeyGlass, RNN, create_time_series_dataset, KL

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

def gather_and_save_predictions(
    name,
    model,
    loader,
    lookback_window,
    bin_edges=None,
    outdir="outputs",
    original_y=None,  # 新增：该 split 的连续真值数组（未离散化）
):
    """
    导出的列：
      - index:            样本原始索引（split 内）
      - y_true:           真实类别索引（离散）
      - y_pred:           预测类别索引（离散，argmax）
      - confidence:       预测类别的 softmax 概率（max 概率）
      - y_true_cont:      真实值的连续近似（其所在区间中心）
      - y_pred_cont:      预测值的连续近似（argmax 对应中心）
      - y_pred_soft_cont: 概率对中心加权的连续近似（∑ p_k * center_k）
      - y_true_raw:       ✅ 原始连续真值（未离散化）
    """
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from pathlib import Path

    model.eval()
    device = next(model.parameters()).device
    rows = []
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # 准备区间中心（与你现有逻辑一致）
    centers = None
    if bin_edges is not None:
        bin_edges = np.asarray(bin_edges)
        if bin_edges.ndim == 1 and len(bin_edges) >= 2:
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    with torch.no_grad():
        for idx, x, y in loader:
            x = x.float().to(device).view(-1, 1, lookback_window)

            logits = model(x)
            probs  = F.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)
            conf   = probs.max(dim=1).values

            preds_np = preds.cpu().numpy()
            conf_np  = conf.cpu().numpy()
            probs_np = probs.cpu().numpy()

            # y -> numpy（离散标签）
            if isinstance(y, torch.Tensor):
                y_np = y.squeeze(-1).cpu().numpy()
            else:
                y_np = np.asarray(y).squeeze()

            # idx -> numpy（split 内索引）
            if isinstance(idx, torch.Tensor):
                idx_np = idx.cpu().numpy()
            else:
                idx_np = np.asarray(idx)

            # 连续近似（中心映射）
            if centers is not None:
                max_c = len(centers) - 1
                y_clipped     = np.clip(y_np,     0, max_c).astype(int)
                preds_clipped = np.clip(preds_np, 0, max_c).astype(int)
                y_true_cont   = centers[y_clipped]
                y_pred_cont   = centers[preds_clipped]

                if probs_np.shape[1] == len(centers):
                    y_pred_soft_cont = (probs_np * centers[np.newaxis, :]).sum(axis=1)
                else:
                    y_pred_soft_cont = y_pred_cont.copy()
            else:
                y_true_cont      = y_np.astype(float)
                y_pred_cont      = preds_np.astype(float)
                y_pred_soft_cont = y_pred_cont.copy()

            # ✅ 原始连续真值（如果提供了 original_y，就用 index 从中取）
            if original_y is not None:
                original_y = np.asarray(original_y)
                # 注意：teacher 有 offset 时，idx 从 offset 开始，这里直接用 idx_np[i] 取就对齐
                y_true_raw_arr = original_y[idx_np.astype(int)]
            else:
                # 没有提供就先用 y_true_cont 占位
                y_true_raw_arr = y_true_cont

            for i in range(len(preds_np)):
                rows.append({
                    "index":            int(idx_np[i]),
                    "y_true":           int(y_np[i]),
                    "y_pred":           int(preds_np[i]),
                    "confidence":       float(conf_np[i]),
                    "y_true_cont":      float(y_true_cont[i]),
                    "y_pred_cont":      float(y_pred_cont[i]),
                    "y_pred_soft_cont": float(y_pred_soft_cont[i]),
                    "y_true_raw":       float(y_true_raw_arr[i]),  # ✅ 真正连续真值
                })

    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    out_path = Path(outdir) / f"{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"[Saved] {out_path}")
    return df



def train_student_model(student_horizon, alpha, num_bins, 
                        val_size, test_size, epochs,
                        temperature, lookback_window, batch_size,
                        patience=5):
    torch.manual_seed(42)
    print(f"\nTraining | Horizon={student_horizon} Alpha={alpha} "
          f"Num bins={num_bins} Val size={val_size} Test size={test_size}"
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
    teacher_train, teacher_val, teacher_test, orig_val_t, orig_test_t, bin_edges_teacher, orig_train_t = create_time_series_dataset(
        data=data, 
        lookback_window=lookback_window, 
        forecasting_horizon=1,
        num_bins=num_bins, 
        val_size=val_size,
        test_size=test_size,
        offset=student_horizon-1, 
        batch_size=batch_size
    )
    # student (H-step, offset=0)
    student_train, student_val, student_test, orig_val_s, orig_test_s, bin_edges_student, orig_train_s = create_time_series_dataset(
        data=data, 
        lookback_window=lookback_window, 
        forecasting_horizon=student_horizon,
        num_bins=num_bins, 
        val_size=val_size,
        test_size=test_size,
        offset=0, 
        batch_size=batch_size
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
            for _, x, y in teacher_val:
                x = x.float().to(device).view(-1,1,lookback_window)
                y = y.float().to(device).squeeze(-1)
                pred = teacher(x).argmax(dim=1).float()
                val_loss += mse(pred, y).item()
            val_loss /= len(teacher_val)

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
            for _, x, y in student_val:
                x = x.float().to(device).view(-1,1,lookback_window)
                y = y.float().to(device).squeeze(-1)
                pred = baseline(x).argmax(dim=1).float()
                val_loss += mse(pred, y).item()
            val_loss /= len(student_val)

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
            for _, x, y in student_val:
                x    = x.float().to(device).view(-1,1,lookback_window)
                pred = student(x).argmax(dim=1).float()
                val_loss += mse(pred, y.float().to(device).squeeze(-1)).item()
            val_loss /= len(student_val)

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


    # ===== 在训练结束后，导出各模型在 train/val/test 上的预测与真实值（含连续值映射） =====
    from pathlib import Path
    outdir = "outputs"
    Path(outdir).mkdir(parents=True, exist_ok=True)

   # Teacher
    gather_and_save_predictions("teacher_train", teacher, teacher_train, lookback_window,
                                bin_edges_teacher, original_y=orig_train_t)
    gather_and_save_predictions("teacher_val",   teacher, teacher_val,   lookback_window,
                                bin_edges_teacher, original_y=orig_val_t)
    gather_and_save_predictions("teacher_test",  teacher, teacher_test,  lookback_window,
                                bin_edges_teacher, original_y=orig_test_t)
    
    # Baseline
    gather_and_save_predictions("baseline_train", baseline, student_train, lookback_window,
                                bin_edges_student, original_y=orig_train_s)
    gather_and_save_predictions("baseline_val",   baseline, student_val,   lookback_window,
                                bin_edges_student, original_y=orig_val_s)
    gather_and_save_predictions("baseline_test",  baseline, student_test,  lookback_window,
                                bin_edges_student, original_y=orig_test_s)
    
    # Student
    gather_and_save_predictions("student_train", student, student_train, lookback_window,
                                bin_edges_student, original_y=orig_train_s)
    gather_and_save_predictions("student_val",   student, student_val,   lookback_window,
                                bin_edges_student, original_y=orig_val_s)
    gather_and_save_predictions("student_test",  student, student_test,  lookback_window,
                                bin_edges_student, original_y=orig_test_s)

                            
    return None

# … rest of your __main__ unchanged …


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Future-Guided Learning for Time-Series Forecasting"
    )
    parser.add_argument("--horizon", type=int, required=True, 
                        help="Student horizon (N)")
    parser.add_argument("--alpha",   type=float, required=True, 
                        help="Loss weight α")
    parser.add_argument("--num_bins",type=int, default=50,   
                        help="Number of bins")
    parser.add_argument("--epochs",  type=int, default=10,   
                        help="Training epochs")
    parser.add_argument("--temperature", type=float, default=4,
                        help='Controls softness of teacher logits')
    parser.add_argument("--lookback_window", type=int, default=1,
                        help="Length of history fed to RNN")
    parser.add_argument("--batch_size",      type=int, default=1,
                        help="Batch size for all loaders")
    parser.add_argument("--val_size",  type=float, default=0.2,
                        help="Fraction of data for validation")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data for testing")

    args = parser.parse_args()

    results = train_student_model(
        student_horizon   = args.horizon,
        alpha             = args.alpha,
        num_bins          = args.num_bins,
        val_size          = args.val_size,
        test_size         = args.test_size,
        epochs            = args.epochs,
        temperature       = args.temperature,
        lookback_window   = args.lookback_window,
        batch_size        = args.batch_size
    )
