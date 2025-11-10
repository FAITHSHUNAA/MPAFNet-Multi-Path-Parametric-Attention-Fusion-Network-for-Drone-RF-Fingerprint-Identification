import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os


def read_signal(csv_file):
    """Read the CSV file line by line"""
    return np.loadtxt(csv_file, delimiter=",")


def compute_stft(signal, n_fft=1024, hop_length=512, win_length=1024, window='hamming'):
    """STFT"""
    return librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)


def compute_file_params(low_file, frame_size=1_000_000, step_size=250_000, num_frames=37):
    """Calculate the maximum amplitude and dB statistical information of a single file"""
    low_freq_signal = read_signal(low_file)

    local_max_amplitude = -np.inf
    local_min_dB = +np.inf
    local_max_dB = -np.inf

    for j in range(num_frames):
        start = j * step_size
        end = start + frame_size
        low_frame = low_freq_signal[start:end]

        # Update the maximum amplitude
        local_max_amplitude = max(local_max_amplitude, np.max(np.abs(low_frame)))

        # calculate STFT
        low_stft = compute_stft(low_frame)

        # calculate dB
        low_db = librosa.amplitude_to_db(np.abs(low_stft), ref=local_max_amplitude)

        # Update dB range
        local_min_dB = min(local_min_dB, np.min(low_db))
        local_max_dB = max(local_max_dB, np.max(low_db))

    return local_max_amplitude, local_min_dB, local_max_dB


def batch_process(low_folder, low_prefix, output_folder, start_idx, end_idx):
    """Batch processing of low-frequency signal files"""
    os.makedirs(output_folder, exist_ok=True)

    for i in range(start_idx, end_idx + 1):
        low_file = os.path.join(low_folder, f"{low_prefix}{i}.csv")

        if os.path.exists(low_file):
            print(f"Processing: {low_file}")

            # Calculate the global parameters of the current file
            global_max_amplitude, global_min_dB, global_max_dB = compute_file_params(low_file)
            print(
                f"File {i}: Max Amplitude: {global_max_amplitude}, Min dB: {global_min_dB}, Max dB: {global_max_dB}")

            # Handle the current file
            process_signal(low_file, output_folder, i, global_max_amplitude, global_min_dB, global_max_dB)
        else:
            print(f"Skipping {i}: File not found.")


def process_signal(low_freq_csv, output_folder, file_index, global_max_amplitude, global_min_dB, global_max_dB):
    """Process low-frequency signals and save the STFT images"""
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
    """Draw and save the STFT image"""
    low_db = librosa.amplitude_to_db(np.abs(low_freq_stft), ref=global_max_amplitude)

    fig, ax = plt.subplots(figsize=(6, 4))
    librosa.display.specshow(low_db, sr=1, hop_length=512, cmap='jet', ax=ax,
                              vmin=global_min_dB, vmax=global_max_dB)

    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
