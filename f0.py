import os
import sys
import numpy as np
import pyworld as pw
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to extract F0 using pyworld
def extract_f0_pyworld(wav_path):
    x, fs = sf.read(wav_path)
    if x.ndim > 1:  # if stereo, take only one channel
        x = x[:, 0]
    _f0, t = pw.dio(x, fs)
    f0 = pw.stonemask(x, _f0, t, fs)
    return f0, t

# base_folder = "/mnt/matylda4/xluner01/F5-TTS/audio_playground/experiments/cz/cz_base_finetuned"

# Check and read base path from command-line
if len(sys.argv) < 2:
    print("Usage: python f0.py <base_folder_path>")
    sys.exit(1)

base_folder = sys.argv[1]

# Dictionary to hold all speaker/experiment/wav paths
data_paths = {}

# Populate data_paths
for speaker in os.listdir(base_folder):
    speaker_path = os.path.join(base_folder, speaker)
    if not os.path.isdir(speaker_path):
        continue

    data_paths[speaker] = {}
    for experiment in os.listdir(speaker_path):
        experiment_path = os.path.join(speaker_path, experiment)
        if not os.path.isdir(experiment_path):
            continue

        wav_paths = [
            os.path.join(experiment_path, f)
            for f in os.listdir(experiment_path)
            if f.endswith(".wav")
        ]

        if wav_paths:
            data_paths[speaker][experiment] = wav_paths

# Function to analyze F0 statistics
def analyze_f0_statistics(wav_paths, plot=False):
    all_means = []
    all_vars = []

    for wav_path in tqdm(wav_paths):
        try:
            f0, t = extract_f0_pyworld(wav_path)
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            continue

        # Extract voiced segments (non-zero F0)
        segments = []
        current_segment = []

        for val in f0:
            if val > 0:
                current_segment.append(val)
            else:
                if len(current_segment) > 1:
                    segments.append(current_segment)
                current_segment = []
        if len(current_segment) > 1:
            segments.append(current_segment)

        # Calculate mean and variance of each segment
        seg_means = [np.mean(seg) for seg in segments]
        seg_vars = [np.var(seg) for seg in segments]

        if seg_means:
            all_means.append(np.mean(seg_means))
            all_vars.append(np.mean(seg_vars))

            if plot:
                plt.figure(figsize=(10, 3))
                plt.plot(t, f0, label="F0")
                plt.xlabel("Time [s]")
                plt.ylabel("F0 [Hz]")
                plt.title(f"F0 contour: {os.path.basename(wav_path)}")
                plt.grid(True)
                plt.legend()
                plt.show()
        # else:
        #     print(f"No voiced segments found in: {wav_path}")

    if all_means:
        return np.mean(all_means), np.mean(all_vars)
    else:
        return 0.0, 0.0

output_file = "f0_statistics.txt"

with open(output_file, "w") as f:
    for speaker, experiments in data_paths.items():
        f.write(f"Analyzing speaker: {speaker}\n")
        print(f"Analyzing speaker: {speaker}")

        # Get max experiment name length for alignment
        max_exp_len = max(len(exp) for exp in experiments.keys())

        for experiment, wavs in experiments.items():
            mean, var = analyze_f0_statistics(wavs, plot=False)
            padded_exp = f"{experiment:<{max_exp_len}}"
            line = f"  {padded_exp}: mean = {mean:.2f}, variance = {var:.2f}"
            f.write(line + "\n")
            print(line)
