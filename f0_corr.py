import numpy as np
import soundfile as sf
import pyworld as pw
from fastdtw import fastdtw
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# base_folder_gen = "/mnt/matylda4/xluner01/F5-TTS/audio_playground/experiments/cz/cz_base_finetuned_f0" # /{babis, nerudova, pavel, schillerova}
# base_folder_gt = "/mnt/matylda4/xluner01/F5-TTS/audio_playground/cz/reference_f0" # /{babis, nerudova, pavel, schillerova}

# Check and read base path from command-line
if len(sys.argv) < 3:
    print("Usage: python f0.py <base_gen_folder_path> <base_gt_folder_path>")
    sys.exit(1)

base_folder_gen = sys.argv[1]
base_folder_gt = sys.argv[2]

# Dictionary to hold all speaker/experiment/wav paths
data_paths_gen = {}
data_paths_gt = {}

# Populate data_paths_gen
for speaker in os.listdir(base_folder_gen):
    speaker_path = os.path.join(base_folder_gen, speaker)
    if not os.path.isdir(speaker_path):
        continue

    data_paths_gen[speaker] = {}
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
            data_paths_gen[speaker][experiment] = wav_paths
            
# Populate data_paths_gt
for speaker in os.listdir(base_folder_gt):
    speaker_path = os.path.join(base_folder_gt, speaker)
    if not os.path.isdir(speaker_path):
        continue

    data_paths_gt[speaker] = {}
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
            data_paths_gt[speaker][experiment] = wav_paths

########################################################################################
def extract_f0_pyworld(wav_path):
    """Extract F0 and time using pyworld."""
    x, fs = sf.read(wav_path)
    _f0, t = pw.dio(x.astype(np.float64), fs)             # Raw F0
    f0 = pw.stonemask(x.astype(np.float64), _f0, t, fs)   # Refined F0
    return f0, t

def dtw_align_f0(f0_1, f0_2):
    """Align F0 vectors using fastdtw."""
    # distance                          = sum of local distances along the optimal warp path
    # path                              = a list of index‐pairs [(i0, j0), (i1, j1), …] describing how frames in f0_1 align to frames in f0_2
    # dist=lambda a, b: abs(a - b)      = the local cost is just the absolute difference of the two F0 values
    distance, path = fastdtw(f0_1, f0_2, dist=lambda a, b: abs(a - b))
    # zip(*path) “unzips” that into two parallel sequence (idx_1 = (i0, i1, i2, …) — the indices in f0_1)
    idx_1, idx_2 = zip(*path)
    aligned_1 = np.array([f0_1[i] for i in idx_1])
    aligned_2 = np.array([f0_2[j] for j in idx_2])
    return aligned_1, aligned_2, path

def plot_warping_path(path):
    """Visualize DTW warping path."""
    idx_1, idx_2 = zip(*path)
    plt.figure(figsize=(6, 6))
    plt.plot(idx_1, idx_2, '.', alpha=0.5)
    plt.title("DTW Warping Path")
    plt.xlabel("Ground Truth F0 Index")
    plt.ylabel("Generated F0 Index")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def filter_unvoiced(f0_1, f0_2):
    """Remove frames where either F0 is zero."""
    mask = (f0_1 > 0) & (f0_2 > 0)
    f0_1_filtered = f0_1[mask]
    f0_2_filtered = f0_2[mask]
    
    assert len(f0_1_filtered) == len(f0_2_filtered), "Filtered F0 vectors have mismatched lengths!"

    return f0_1[mask], f0_2[mask]

def compute_f0_correlation(f0_gt, f0_gen):
    """Compute Pearson correlation between two F0 vectors."""
    if len(f0_gt) < 2 or len(f0_gen) < 2:
        return np.nan
    return pearsonr(f0_gt, f0_gen)[0]

def evaluate_f0_consistency(gt_wav, gen_wav, plot_warping_path=False):
    """Full pipeline to evaluate F0 consistency."""
    f0_gt, _ = extract_f0_pyworld(gt_wav)
    f0_gen, _ = extract_f0_pyworld(gen_wav)
    
    f0_gt_aligned, f0_gen_aligned, path = dtw_align_f0(f0_gt, f0_gen)
    if plot_warping_path:
      plot_warping_path(path)

    # corr = compute_f0_correlation(f0_gt_aligned, f0_gen_aligned)
    # print(f"F0 Pearson correlation (all frames): {corr:.4f}")
    
    f0_gt_filtered, f0_gen_filtered = filter_unvoiced(f0_gt_aligned, f0_gen_aligned)
    
    corr = compute_f0_correlation(f0_gt_filtered, f0_gen_filtered)
    print(f"F0 Pearson correlation (voiced frames only): {corr:.2f}")

    return corr
########################################################################################
output_file = "f0_stats.txt"

with open(output_file, "w") as outf:
    # iterate over speakers present in both generated and GT sets
    for speaker in sorted(set(data_paths_gen) & set(data_paths_gt)):
        outf.write(f"{speaker}\n")
        print(f"Analyzing speaker: {speaker}")

        exps_gen = data_paths_gen[speaker]
        exps_gt  = data_paths_gt[speaker]

        # compute max experiment name width for nice alignment
        all_exps = sorted(set(exps_gen) & set(exps_gt))
        max_exp_len = max(len(exp) for exp in all_exps)

        for exp in all_exps:
            gen_wavs = sorted(exps_gen[exp])
            gt_wavs  = sorted(exps_gt[exp])

            # build a map from basename -> gt path for quick lookup
            gt_map = {os.path.basename(p): p for p in gt_wavs}

            corrs = []
            for gen_path in tqdm(gen_wavs):
                name = os.path.basename(gen_path)
                if name not in gt_map:
                    print(f"  [!] Missing GT for {speaker}/{exp}/{name}, skipping.")
                    continue

                gt_path = gt_map[name]
                corr = evaluate_f0_consistency(gt_path, gen_path, plot_warping_path=False)
                corrs.append(corr)

            if not corrs:
                line = f"  {exp:<{max_exp_len}}: no matching files found\n"
                outf.write(line)
                print(line, end="")
                continue

            mean_corr = float(np.mean(corrs))
            # format individual correlations to two decimal places
            corr_strs = ", ".join(f"{c:.2f}" for c in corrs)

            line = (
                f"  {exp:<{max_exp_len}}: "
                f"mean = {mean_corr:.2f} "
                f"({corr_strs})\n"
            )
            outf.write(line)
            print(line, end="")

print(f"\nResults written to {output_file}")
