import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os


def read_signal(csv_file):
    """按行读取 CSV 文件"""
    return np.loadtxt(csv_file, delimiter=",")


def compute_stft(signal, n_fft=1024, hop_length=512, win_length=1024, window='hamming'):
    """计算 STFT 变换"""
    return librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)


def compute_file_params(low_file, frame_size=1_000_000, step_size=250_000, num_frames=37):
    """计算单个文件的最大幅值和 dB 统计信息（只用低频）"""
    low_freq_signal = read_signal(low_file)

    local_max_amplitude = -np.inf
    local_min_dB = +np.inf
    local_max_dB = -np.inf

    for j in range(num_frames):
        start = j * step_size
        end = start + frame_size
        low_frame = low_freq_signal[start:end]

        # 更新最大振幅
        local_max_amplitude = max(local_max_amplitude, np.max(np.abs(low_frame)))

        # 计算 STFT
        low_stft = compute_stft(low_frame)

        # 计算 dB
        low_db = librosa.amplitude_to_db(np.abs(low_stft), ref=local_max_amplitude)

        # 更新 dB 范围
        local_min_dB = min(local_min_dB, np.min(low_db))
        local_max_dB = max(local_max_dB, np.max(low_db))

    return local_max_amplitude, local_min_dB, local_max_dB


def batch_process(low_folder, low_prefix, output_folder, start_idx, end_idx):
    """批量处理低频信号文件"""
    os.makedirs(output_folder, exist_ok=True)

    for i in range(start_idx, end_idx + 1):
        low_file = os.path.join(low_folder, f"{low_prefix}{i}.csv")

        if os.path.exists(low_file):
            print(f"Processing: {low_file}")

            # 计算当前文件的全局参数
            global_max_amplitude, global_min_dB, global_max_dB = compute_file_params(low_file)
            print(
                f"File {i}: Max Amplitude: {global_max_amplitude}, Min dB: {global_min_dB}, Max dB: {global_max_dB}")

            # 处理当前文件
            process_signal(low_file, output_folder, i, global_max_amplitude, global_min_dB, global_max_dB)
        else:
            print(f"Skipping {i}: File not found.")


def process_signal(low_freq_csv, output_folder, file_index, global_max_amplitude, global_min_dB, global_max_dB):
    """处理低频信号并保存 STFT 图像"""
    os.makedirs(output_folder, exist_ok=True)

    low_freq_signal = read_signal(low_freq_csv)
    frame_size = 1_000_000
    step_size = 250_000
    num_frames = (len(low_freq_signal) - frame_size) // step_size + 1

    for i in range(num_frames):
        start = i * step_size
        end = start + frame_size
        low_frame = low_freq_signal[start:end]
        low_stft = compute_stft(low_frame)

        save_path = os.path.join(output_folder, f"{file_index}_{i}.png")
        plot_and_save_stft(low_stft, save_path, global_max_amplitude, global_min_dB, global_max_dB)
        print(f"Saved: {save_path}")


def plot_and_save_stft(low_freq_stft, save_path, global_max_amplitude, global_min_dB, global_max_dB):
    """绘制并保存 STFT 图像（ECSG 能量校准）"""
    low_db = librosa.amplitude_to_db(np.abs(low_freq_stft), ref=global_max_amplitude)

    fig, ax = plt.subplots(figsize=(6, 4))
    librosa.display.specshow(low_db, sr=1, hop_length=512, cmap='jet', ax=ax,
                              vmin=global_min_dB, vmax=global_max_dB)

    plt.axis("off")  # 去掉坐标轴
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    # 设置参数
    low_folder = r"E:\2019-DroneRF 数据集：用于基于 RF 的检测、分类和识别的无人机数据集\DroneRF\Phantom drone\RF Data_11000_H"
    low_prefix = "11000H_"
    output_folder = r"F:\DroneRF\四路特征\high_img_9"
    start_idx = 0
    end_idx = 20

    # 批量处理
    batch_process(low_folder, low_prefix, output_folder, start_idx, end_idx)




# 检查数据
# import re
#
# # 读取 CSV 文件
# file_path = r"E:\DroneRF\DroneRF\Bepop drone\RF Data_10000_L\RF Data_10000_L\10000L_19.csv"  # 修改为你的文件路径
# with open(file_path, 'r', encoding='utf-8') as f:
#     line = f.readline().strip()
#
# # 定义正则表达式，匹配合法数字（整数或小数）
# def is_valid_number(value):
#     return bool(re.fullmatch(r'\d+(\.\d+)?', value))
#
# # 处理数据
# values = line.split(',')  # CSV 逗号分隔
# corrected_values = [val if is_valid_number(val) else '0' for val in values]
#
# # 保存修改后的数据
# with open(file_path, 'w', encoding='utf-8') as f:
#     f.write(','.join(corrected_values))
