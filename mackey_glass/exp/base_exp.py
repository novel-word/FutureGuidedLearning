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
    outdir="outputs"
):
    """
    收集一个数据划分(split)上的预测与真实值，并保存到 CSV。

    导出的列：
      - index:       样本原始索引（DataLoader 返回的第一个元素）
      - y_true:      真实类别索引（离散）
      - y_pred:      预测类别索引（离散，argmax）
      - confidence:  预测类别的 softmax 概率
      - y_true_cont: 真实值的连续近似（其所在区间的中心）
      - y_pred_cont: 预测值的连续近似（预测区间的中心）

    参数：
      name:            文件名前缀，例如 'student_test'
      model:           要评估的模型
      loader:          对应划分的 DataLoader，迭代产出 (idx, x, y)
      lookback_window: RNN 期望的历史长度，用于 reshape
      bin_edges:       分箱边界（长度 = num_bins-1 或 num_bins+1，与你的实现一致）
      outdir:          输出目录
    """
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F

    model.eval()
    device = next(model.parameters()).device
    rows = []

    Path(outdir).mkdir(parents=True, exist_ok=True)

    # 若提供了 bin_edges，计算区间中心；并做一次性缓存
    centers = None
    if bin_edges is not None:
        bin_edges = np.asarray(bin_edges)
        # 兼容 np.linspace 产生的 (num_bins-1) 边界或 (num_bins+1) 边界两种写法
        if len(bin_edges) >= 2:
            # 若是 (num_bins-1) 边界（不含两端），需要补上下界；用端点外推一点点
            if len(bin_edges.shape) == 1 and (np.diff(bin_edges) > 0).all():
                # 简单鲁棒补边（只在像 utils 中那样给出 num_bins-1 时触发）
                if len(bin_edges) == 0:
                    centers = None
                elif len(bin_edges) >= 2 and (np.isfinite(bin_edges).all()):
                    # 补上下界：用首/尾间距外推
                    d_lo  = bin_edges[1] - bin_edges[0]
                    d_hi  = bin_edges[-1] - bin_edges[-2]
                    full_edges = np.concatenate([
                        [bin_edges[0] - d_lo],
                        bin_edges,
                        [bin_edges[-1] + d_hi]
                    ])
                    centers = 0.5 * (full_edges[:-1] + full_edges[1:])
                else:
                    centers = None
            # 若本来就是 (num_bins+1) 个边界，直接取中心
            if centers is None and len(bin_edges) >= 2:
                centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    with torch.no_grad():
        for idx, x, y in loader:
            # x: [B, lookback] -> [B, 1, lookback]
            x = x.float().to(device).view(-1, 1, lookback_window)

            logits = model(x)                    # [B, num_bins]
            probs  = F.softmax(logits, dim=1)    # [B, num_bins]
            preds  = probs.argmax(dim=1)         # [B]
            conf   = probs.max(dim=1).values     # [B]

            # 整理为 CPU numpy
            preds_np = preds.cpu().numpy()
            conf_np  = conf.cpu().numpy()

            # y 的形状通常是 [B, 1] 或 [B]，这里统一 squeeze
            if isinstance(y, torch.Tensor):
                y_np = y.squeeze(-1).cpu().numpy()
            else:
                y_np = np.asarray(y).squeeze()

            # 处理 idx：可能是 tensor/list/ndarray
            if isinstance(idx, torch.Tensor):
                idx_np = idx.cpu().numpy()
            else:
                idx_np = np.asarray(idx)

            # 连续值映射（若 centers 有效）
            if centers is not None:
                # 为防止极端越界，做一下 clip
                max_c = len(centers) - 1
                y_clipped     = np.clip(y_np,     0, max_c).astype(int)
                preds_clipped = np.clip(preds_np, 0, max_c).astype(int)
                y_true_cont   = centers[y_clipped]
                y_pred_cont   = centers[preds_clipped]
            else:
                # 若没有 bin_edges，则用离散索引本身占位
                y_true_cont = y_np.astype(float)
                y_pred_cont = preds_np.astype(float)

            # 逐条写入
            for i in range(len(preds_np)):
                rows.append({
                    "index":       int(idx_np[i]),
                    "y_true":      int(y_np[i]),
                    "y_pred":      int(preds_np[i]),
                    "confidence":  float(conf_np[i]),
                    "y_true_cont": float(y_true_cont[i]),
                    "y_pred_cont": float(y_pred_cont[i]),
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
    teacher_train, teacher_val, teacher_test, _, _, bin_edges_teacher = create_time_series_dataset(
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
    student_train, student_val, student_test, _, _, bin_edges_student = create_time_series_dataset(
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

    # Teacher 使用 teacher_* 的 loader + bin_edges_teacher
    gather_and_save_predictions(
        name="teacher_train",
        model=teacher,
        loader=teacher_train,
        lookback_window=lookback_window,
        bin_edges=bin_edges_teacher,
        outdir=outdir
    )
    gather_and_save_predictions(
        name="teacher_val",
        model=teacher,
        loader=teacher_val,
        lookback_window=lookback_window,
        bin_edges=bin_edges_teacher,
        outdir=outdir
    )
    gather_and_save_predictions(
        name="teacher_test",
        model=teacher,
        loader=teacher_test,
        lookback_window=lookback_window,
        bin_edges=bin_edges_teacher,
        outdir=outdir
    )

    # Baseline 与 Student 共用 student_* 的 loader；分箱与任务一致，传 bin_edges_student
    gather_and_save_predictions(
        name="baseline_train",
        model=baseline,
        loader=student_train,
        lookback_window=lookback_window,
        bin_edges=bin_edges_student,
        outdir=outdir
    )
    gather_and_save_predictions(
        name="baseline_val",
        model=baseline,
        loader=student_val,
        lookback_window=lookback_window,
        bin_edges=bin_edges_student,
        outdir=outdir
    )
    gather_and_save_predictions(
        name="baseline_test",
        model=baseline,
        loader=student_test,
        lookback_window=lookback_window,
        bin_edges=bin_edges_student,
        outdir=outdir
    )

    gather_and_save_predictions(
        name="student_train",
        model=student,
        loader=student_train,
        lookback_window=lookback_window,
        bin_edges=bin_edges_student,
        outdir=outdir
    )
    gather_and_save_predictions(
        name="student_val",
        model=student,
        loader=student_val,
        lookback_window=lookback_window,
        bin_edges=bin_edges_student,
        outdir=outdir
    )
    gather_and_save_predictions(
        name="student_test",
        model=student,
        loader=student_test,
        lookback_window=lookback_window,
        bin_edges=bin_edges_student,
        outdir=outdir
    )
                            
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
