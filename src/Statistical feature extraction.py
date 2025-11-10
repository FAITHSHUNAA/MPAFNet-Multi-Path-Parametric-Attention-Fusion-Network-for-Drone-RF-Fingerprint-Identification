import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks, hilbert, stft
import pywt
import os
from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==== Feature extraction function ====
def envelope_mean(signal):
    analytic_signal = hilbert(signal.astype(np.float32))
    return np.mean(np.abs(analytic_signal))

def high_order_skewness(signal):
    return np.mean((signal - np.mean(signal))**3) / (np.std(signal)**3 + 1e-10)


def autocorr_peak(signal):
    n = len(signal)
    autocorr = np.correlate(signal, signal, mode='full')[n-1:n+n//4]
    peaks, _ = find_peaks(autocorr)
    return np.max(autocorr[peaks]) / (autocorr[0] + 1e-10) if len(peaks) > 0 else 0

def freq_centroid(signal, fs=40e6):
    freqs, psd = welch(signal, fs=fs, nperseg=256)
    return np.sum(freqs * psd) / (np.sum(psd) + 1e-10)

def mod_bandwidth(signal, fs=40e6):
    freqs, psd = welch(signal, fs=fs, nperseg=256)
    threshold = 0.05 * np.max(psd)
    return np.sum(psd > threshold) * (fs / len(psd))

def cumulant_c42(signal):
    m2 = np.mean(signal**2)
    m4 = np.mean(signal**4)
    return m4 - 3 * m2**2

def freq_hop_entropy(signal, fs=40e6):
    _, psd = welch(signal, fs=fs, nperseg=256)
    psd = psd / (np.sum(psd) + 1e-10)
    return -np.sum(psd * np.log2(psd))

def sparse_tf_peak(signal, fs=40e6, nperseg=256):
    _, _, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    return np.max(np.abs(Zxx))

def wavelet_entropy(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    entropy = 0
    for coeff in coeffs:
        abs_coeff = np.abs(coeff)
        coeff_sum = np.sum(abs_coeff)
        if coeff_sum < 1e-10:
            continue
        p = abs_coeff / coeff_sum + 1e-10
        p = np.maximum(p, 1e-10)
        entropy -= np.sum(p * np.log2(p))
    return entropy

def inst_freq_mean(signal, fs=40e6):
    analytic_signal = hilbert(signal.astype(np.float32))
    inst_phase = np.unwrap(np.angle(analytic_signal))
    inst_freq = np.diff(inst_phase) * (fs / (2.0 * np.pi))
    return np.mean(np.abs(inst_freq))

def tf_sparsity(signal, fs=40e6, nperseg=256):
    _, _, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    return np.sum(np.abs(Zxx)) / (np.sqrt(np.sum(np.abs(Zxx)**2)) + 1e-10)

def extract_features(signal):
    return [
        envelope_mean(signal),
        high_order_skewness(signal),
        autocorr_peak(signal),
        freq_centroid(signal),
        mod_bandwidth(signal),
        cumulant_c42(signal),
        freq_hop_entropy(signal),
        sparse_tf_peak(signal),
        wavelet_entropy(signal),
        inst_freq_mean(signal),
        tf_sparsity(signal)
    ]

# ==== Single-file processing function ====
def process_file(input_path, output_path):
    try:
        raw_data = np.loadtxt(input_path, delimiter=',', dtype=np.float32)
        frame_length = 1_000_000
        step = int(frame_length * 0.25)
        num_frames = (len(raw_data) - frame_length) // step + 1

        all_features = []
        for i in range(num_frames):
            start_idx = i * step
            end_idx = start_idx + frame_length
            frame = raw_data[start_idx:end_idx]
            features = extract_features(frame)
            all_features.append(features)

        feature_names = [
            'env_mean', 'skewness', 'pulse_rate', 'autocorr_peak',
            'freq_centroid', 'mod_bandwidth', 'cumulant_c42', 'freq_hop_entropy',
            'sparse_tf_peak', 'wavelet_entropy', 'inst_freq_mean', 'tf_sparsity'
        ]
        df_features = pd.DataFrame(all_features, columns=feature_names)
        df_features.to_csv(output_path, index=False)
        return f"{os.path.basename(input_path)} 完成，共 {num_frames} 帧"
    except Exception as e:
        return f"{os.path.basename(input_path)} 失败: {e}"

# ==== Multi-process batch processing ====
def process_all_files_parallel(input_dir, output_dir, base_filename, start_index=0, max_index=1000, max_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[{datetime.now()}] 开始并行处理...")

    total_start = time.time()
    tasks = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx in range(start_index, max_index + 1):
            filename = f"{base_filename}_{idx}.csv"
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(input_path):
                tasks.append(executor.submit(process_file, input_path, output_path))
            else:
                print(f"[{datetime.now()}] 跳过不存在文件: {filename}")

        for future in as_completed(tasks):
            print(f"[{datetime.now()}] {future.result()}")

    print(f"[{datetime.now()}] 所有任务完成，总用时 {time.time() - total_start:.2f} 秒")
