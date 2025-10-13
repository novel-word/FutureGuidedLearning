# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample

# ---- NumPy 2.x 兼容垫片：为依赖旧别名的库恢复 numpy.lib.pad ----
if not hasattr(np.lib, "pad"):
    np.lib.pad = np.pad


import stft

from utils.group_seizure_teacher import group_seizure
from utils.log import log
from utils.save_load import (
    save_pickle_file, load_pickle_file,
    save_hickle_file, load_hickle_file
)

# --------------------------- 工具函数 ---------------------------

def make_teacher(mode, teacher_settings, shuffle=False):
    """
    生成教师模型所需的 (ictal/interictal) 数据：
    返回：
      ictal_data_X, ictal_data_y, interictal_data_X, interictal_data_y
    其中：
      - ictal_data_X / ictal_data_y 是按“发作段/正样本”分组后的列表
      - interictal_data_X / interictal_data_y 是负样本（整合为一个 X,y）
    """
    import random

    def shuffle_lists(list1, list2):
        combined = list(zip(list1, list2))
        random.shuffle(combined)
        a, b = zip(*combined)
        return list(a), list(b)

    dog_targets   = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4']
    human_targets = ['Patient_3', 'Patient_5', 'Patient_6', 'Patient_7']

    ictal_data_X, ictal_data_y = [], []
    interictal_data_X, interictal_data_y = [], []

    if mode == 'Dog':
        freq = 200
        targets = dog_targets
        teacher_channels = None
    elif mode == 'Patient_1':
        # Kaggle 人类数据，采样率 1000Hz，教师模型用较少通道
        freq = 1000
        targets = human_targets
        teacher_channels = 15
    else:
        # Patient_2 模式：同样 1000Hz，但用更多通道
        freq = 1000
        targets = human_targets
        teacher_channels = 24

    for target in targets:
        ictal_X, ictal_y = PrepDataTeacher(
            target, type='ictal',
            settings=teacher_settings, freq=freq,
            teacher_channels=teacher_channels
        ).apply()

        interictal_X, interictal_y = PrepDataTeacher(
            target, type='interictal',
            settings=teacher_settings, freq=freq,
            teacher_channels=teacher_channels
        ).apply()

        # 发作段（正样本）按段追加（Xg/yg 是 list）
        ictal_data_X.extend(ictal_X)
        ictal_data_y.extend(ictal_y)

        # 负样本通常是一个整体 X,y；为了与旧接口兼容，按“每个 target 追加一对”
        interictal_data_X.append(interictal_X)
        interictal_data_y.append(interictal_y)

    if shuffle:
        ictal_data_X, ictal_data_y = shuffle_lists(ictal_data_X, ictal_data_y)

    return ictal_data_X, ictal_data_y, interictal_data_X, interictal_data_y


def makedirs(dir_path: str):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception:
        pass

def shuffle_lists(list1, list2):
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    list1[:], list2[:] = zip(*combined)
    return list1, list2

# ---------------------- 通道重要性（保持原样） -------------------

# channel significance order of the teacher model for each patient
global significance
significance = {
    'Patient_1':[17,46,8,28,25,16,18,9,24,19,12,42,11,7,23,0,32,45,3,2,4,14,21,27,34],
    'Patient_2':None,
    'Patient_3':[3,4,43,24,0,18,10,5,6,46,25,44,9,1,8,23,17,15,11,38,35,2,22,42,14],
    'Patient_4':[35,43,42,44,34,40,2,32,0,36,1,45,31,28,12,20,37,33,3,14,21,23,46,39,7],
    'Patient_5':[0,16,8,7,23,13,5,4,15,22,1,12,30,6,21,28,17,3,39,9,19,26,29,33,14],
    'Patient_6':[13,22,21,14,23,5,15,6,12,28,3,2,20,9,27,19,4,1,16,11,24,10,17,8,7],
    'Patient_7':[26,7,34,10,8,24,9,29,28,30,25,31,33,32,27,19,23,4,22,13,17,11,18,6,21],
    'Patient_8':None
}

# ---------------------- MATLAB 结构体兼容工具 --------------------

def _safe_get_struct_field(mat_struct, field_name):
    """
    兼容 scipy.io.loadmat 读出的 MATLAB struct（void/ndarray 包壳）
    返回 numpy 数组或标量；自动去掉 (1,1) 的外壳
    """
    if isinstance(mat_struct, dict):
        v = mat_struct[field_name]
    else:
        # 结构体对象：直接 getattr
        v = getattr(mat_struct, field_name)
    # 逐层剥壳
    while isinstance(v, np.ndarray) and v.size == 1:
        v = v.item()
    return v

def _extract_segment_from_mat(mat_dict):
    """
    兼容两种格式：
    A. 顶层直接有 'data' / 'sampling_frequency' / 'channels'
    B. 顶层只有一个类似 'preictal_segment_1' 的结构体，字段在里面
    返回：data_arr, sampling_freq(或None), channels(或None), latency(或None)
    """
    # A. 顶层字段直出
    if 'data' in mat_dict:
        data_arr = mat_dict['data']
        sf = mat_dict.get('sampling_frequency', None)
        ch = mat_dict.get('channels', None)
        lat = mat_dict.get('latency', None)
        # 把 ndarray 的单元素变成标量
        if isinstance(sf, np.ndarray) and sf.size == 1:
            sf = sf.item()
        if isinstance(lat, np.ndarray) and lat.size == 1:
            lat = lat.item()
        return data_arr, sf, ch, lat

    # B. 找出不以下划线开头的 key（排除 __header__/__globals__/__version__）
    keys = [k for k in mat_dict.keys() if not k.startswith('__')]
    if len(keys) == 0:
        raise ValueError("MAT file has no usable keys.")
    seg = mat_dict[keys[0]]
    # 可能是 (1,1) ndarray 包着 struct
    if isinstance(seg, np.ndarray) and seg.size == 1:
        seg = seg.item()

    data_arr = _safe_get_struct_field(seg, 'data')
    sf = None
    ch = None
    lat = None
    # 可选字段安全获取
    for name in ('sampling_frequency', 'channels', 'latency'):
        try:
            val = _safe_get_struct_field(seg, name)
        except Exception:
            val = None
        if name == 'sampling_frequency':
            sf = val
        elif name == 'channels':
            ch = val
        elif name == 'latency':
            lat = val
    # 单元素标量化
    if isinstance(sf, np.ndarray) and sf.size == 1:
        sf = sf.item()
    if isinstance(lat, np.ndarray) and lat.size == 1:
        lat = lat.item()
    return data_arr, sf, ch, lat

# -------------------------- 主类 -------------------------------

class PrepDataTeacher():
    def __init__(self, target, type, settings, freq, teacher_channels=None):
        self.target = target
        self.settings = settings
        self.type = type                  # 'ictal' / 'interictal' / 'preictal'
        self.freq = freq                  # 目标采样率（重采样目标）
        self.teacher_channels = teacher_channels

    def most_significant_channels(self, data, channels, num_channels):
        channels = channels[0:num_channels]
        result_matrix = data[channels, :]
        return result_matrix

    # ---------------------- 关键修改函数 ------------------------
    def load_signals_Kaggle2014Det(self):
        """
        返回：
          result: [np.ndarray(channel, samples), ...] 逐文件的数据列表
          latencies: List[int] 按“累计样本点”的边界（首部含 0，末尾含总长），
                     若 .mat 内含 latency 字段，则保持原逻辑；否则用“每个文件末尾”为边界。
        """
        data_dir = self.settings['datadir']
        print(f'Seizure Detection - Loading {self.type} data for patient {self.target}')
        dir_path = os.path.join(data_dir, self.target)

        done = False
        i = 0
        result = []

        # 我们依旧返回 latencies，但当没有 latency 字段时，
        # 用“累计样本长度”的文件末尾来代替（即 boundaries）
        latencies = [0]

        prev_latency = -1
        targetFrequency = self.freq

        while not done:
            i += 1
            # 兼容两种命名：Patient_X_type_segment_i.mat 与 type_segment_i.mat
            candidates = [
                f'{dir_path}/{self.target}_{self.type}_segment_{i}.mat',
                f'{dir_path}/{self.type}_segment_{i}.mat'
            ]
            filename = None
            for cand in candidates:
                if os.path.exists(cand):
                    filename = cand
                    break

            if not filename:
                done = True
                continue

            # 使用 squeeze_me 简化结构体外壳
            mat = loadmat(filename, squeeze_me=True, struct_as_record=False)

            # 兼容不同顶层结构（含/不含 latency）
            data_arr, file_sf, file_channels, file_latency = _extract_segment_from_mat(mat)
            # data_arr 期望形状：(n_channels, n_samples)

            # 可选的通道筛选（与原逻辑保持）
            if "Patient_" in self.target and self.teacher_channels is not None and significance.get(self.target):
                ch_order = significance[self.target]
                data_arr = self.most_significant_channels(data_arr, ch_order, self.teacher_channels)

            # 采样率：优先用文件里带的 sampling_frequency；没有就默认认为已经是目标频率
            cur_freq = None
            if file_sf is not None:
                try:
                    cur_freq = int(file_sf)
                except Exception:
                    cur_freq = None

            # 统一重采样至 targetFrequency（按最后一维做）
            if cur_freq is not None and cur_freq != targetFrequency:
                n_samples_new = int(round(data_arr.shape[-1] * targetFrequency / float(cur_freq)))
                data_arr = resample(data_arr, n_samples_new, axis=data_arr.ndim - 1)

            # 记录本段
            seg_len = data_arr.shape[-1]
            result.append(data_arr)

            # latency 处理：
            # - 如果文件里有 latency（历史 Kaggle2014Det 逻辑），沿用原判定方式；
            # - 否则，把“每个文件末尾”的累计样本点作为边界。
            if file_latency is not None:
                try:
                    latency = int(file_latency)
                except Exception:
                    # 有些情况下 latency 可能是浮点或数组
                    latency = int(np.round(float(file_latency)))
                # 原逻辑：latency 回绕（下降）时，说明进入下一段发作
                if prev_latency >= 0 and latency < prev_latency:
                    latencies.append(i * targetFrequency)
                prev_latency = latency
            else:
                # 用累计长度边界
                # 注意：latencies 里存的是“累计样本点”，“i*targetFrequency”在无 latency 时不可靠
                # 这里不立即 append，等循环结束后统一把所有文件末尾累计点加入
                pass

        # 收尾：补一个“总长”边界
        # 如果有真实 latency 判断过（prev_latency 改变过），保持原始 i*targetFrequency 边界；
        # 否则用累计样本法：
        if prev_latency >= 0:
            # 延续原文件的末尾边界（保持兼容）
            latencies.append(i * targetFrequency)
        else:
            # 累计样本边界：从 result 的长度累加
            cum = 0
            # 先清空仅有的 [0]，改为 [0, 每个文件末尾的累计点 ...]
            latencies = [0]
            for seg in result:
                cum += seg.shape[-1]
                latencies.append(cum)

        print(latencies)
        return result, latencies

    # ---------------------- 其余保持原样 ------------------------

    @staticmethod
    def combine_matrices(matrix_list):
        if not matrix_list:
            raise ValueError("Matrix list is empty.")
        num_rows = matrix_list[0].shape[0]
        if not all(matrix.shape[0] == num_rows for matrix in matrix_list):
            raise ValueError("Number of rows in all matrices must be the same.")
        combined_matrix = np.concatenate(matrix_list, axis=1)
        # 原文件风格：转为 (time, channels) 便于后续滑窗
        return np.transpose(combined_matrix)

    def process_raw_data(self):
        """
        按原有代码风格保留接口与主要流程（STFT + 滑窗）。
        说明：
        - 当没有 latency 时，我们返回的 latencies 实际上是“文件末尾”的累计样本点；
          后续的 onset_indices 命中逻辑依旧可用。
        """
        result, latencies = self.load_signals_Kaggle2014Det()
        combination = PrepDataTeacher.combine_matrices(result)  # (time, channels)

        X_data, y_data = [], []

        # 预处理参数（保持原有经验值）
        targetFrequency = self.freq
        if 'Patient_' in self.target:
            DataSampleSize = int(targetFrequency / 5)
        else:
            DataSampleSize = targetFrequency
        numts = 30  # 每个样本包含的时间窗数量（和原代码一致）

        # 采样策略：根据类型设置窗口与步长
        if self.type == 'ictal':
            # 正样本：较密集
            win_len = DataSampleSize
            step = int(win_len / 2)
            divisor = int((numts * win_len) / step)  # 用于周期性打标签（1/2）
            onset_indices = []

            i = 0
            for a in range(0, combination.shape[0] - numts * win_len + 1, step):
                b = a + numts * win_len
                s = combination[a:b, :]  # (time, channels)

                # 频谱特征
                stft_data = stft.spectrogram(s, framelength=DataSampleSize, centered=False)
                stft_data = stft_data[1:, :, :]  # 去掉 DC 分量
                stft_data = np.log10(stft_data)
                stft_data[stft_data <= 0] = 0
                stft_data = np.transpose(stft_data, (2, 1, 0))  # (T, C, F)
                stft_data = np.abs(stft_data) + 1e-6
                stft_data = stft_data.reshape(-1, 1, stft_data.shape[0], stft_data.shape[1], stft_data.shape[2])

                X_data.append(stft_data)
                # 标签：与原逻辑保持一致，周期性地打 1/2
                y_data.append(1 if (i % divisor == 0 or i == 0) else 2)

                # 命中边界：b 是否落在 latencies 中（无 latency 时即文件边界）
                if b in latencies:
                    onset_indices.append(i)
                i += 1

            onset_indices.append(i)

            Xg, yg = group_seizure(X=X_data, y=y_data, onset_indices=onset_indices)
            print('Number of seizures %d' % len(Xg), Xg[0].shape, yg[0].shape)
            return Xg, yg

        elif self.type == 'interictal':
            # 负样本：步长更大
            win_len = DataSampleSize
            step = int(win_len)  # 非重叠或小重叠
            divisor = int((numts * win_len) / step)

            i = 0
            for a in range(0, combination.shape[0] - numts * win_len + 1, step):
                b = a + numts * win_len
                s = combination[a:b, :]

                stft_data = stft.spectrogram(s, framelength=DataSampleSize, centered=False)
                stft_data = stft_data[1:, :, :]
                stft_data = np.log10(stft_data)
                stft_data[stft_data <= 0] = 0
                stft_data = np.transpose(stft_data, (2, 1, 0))
                stft_data = np.abs(stft_data) + 1e-6
                stft_data = stft_data.reshape(-1, 1, stft_data.shape[0], stft_data.shape[1], stft_data.shape[2])

                X_data.append(stft_data)
                # 负样本：0 / -1（保持原逻辑）
                y_data.append(0 if (i % divisor == 0 or i == 0) else -1)
                i += 1

            X = np.concatenate(X_data)
            y = np.array(y_data)
            print('X', X.shape, 'y', y.shape)
            return X, y

        else:
            raise ValueError(f"Unknown type: {self.type}")

    def apply(self):
        """
        缓存与落盘逻辑保持原样（如果你原项目已有缓存，这里可按需改写/接入）。
        这里直接计算返回，保持与旧接口兼容。
        """
        print(f'Preparing {self.type} data for {self.target} ...')
        return self.process_raw_data()


# ------------------------- 例子：批量加载 -------------------------
# 你可以继续保留/扩展原文件中面向项目的批量入口（如需要）
# 本文件的目标是修复 MAT 读取与无 latency 情况下的边界定义。
#
# 如果你原工程在别处调用了：
#   ictal_X, ictal_y = PrepDataTeacher(target, type='ictal', settings=..., freq=..., teacher_channels=...).apply()
#   interictal_X, interictal_y = PrepDataTeacher(target, type='interictal', settings=..., freq=..., teacher_channels=...).apply()
# 上面的接口保持不变，可直接替换文件使用。
