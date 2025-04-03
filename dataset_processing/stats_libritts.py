import os
import csv
import argparse
import librosa
from tqdm import tqdm

def calculate_stats(dataset_path):
    num_samples = 0
    total_duration = 0.0
    total_chars = 0
    
    print(f"Calculating statistics for dataset: {dataset_path}")
    
    for root, _, files in os.walk(dataset_path):
        trans_files = [f for f in files if f.endswith(".trans.tsv")]
        
        for trans_file in trans_files:
            trans_path = os.path.join(root, trans_file)
            
            with open(trans_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if row and len(row) >= 3:
                        utterance_id, original_text, normalized_text = row[:3]
                        
                        wav_path = os.path.join(root, utterance_id + ".wav")
                        if os.path.exists(wav_path):
                            duration = librosa.get_duration(path=wav_path)
                            num_samples += 1
                            total_duration += duration
                            total_chars += len(normalized_text)
    
    avg_duration = total_duration / num_samples if num_samples > 0 else 0
    avg_chars = total_chars / num_samples if num_samples > 0 else 0
    
    print(f"Dataset: {dataset_path}")
    print(f"Number of samples: {num_samples}")
    print(f"Average duration (seconds): {avg_duration:.2f}")
    print(f"Average number of characters in transcription: {avg_chars:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to the dataset folder (e.g., train-clean-100)")
    args = parser.parse_args()
    
    calculate_stats(args.dataset_path)
