import jiwer
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from jiwer import wer, cer
import argparse
from tqdm import tqdm
import torch
import gc
import whisper

WER_VALUES = []
CER_VALUES = []
WHISPER_MODEL = None
WHISPER_MODEL_DIR = "./pretrained_models/whisper"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# python wer_cer.py \
# --generated_audio_folder "/mnt/matylda4/xluner01/F5-TTS/audio_playground/experiments/cz/non_causal/babis" \
# --reference_transcriptions_folder "/mnt/matylda4/xluner01/F5-TTS/audio_playground/cz/to_generate" \
# --output_folder_plots "/mnt/matylda4/xluner01/F5-TTS/audio_playground/experiments/cz/non_causal/babis" \
# --plot_title "Czech non-causal unseen male speaker" \
# --output_folder_results "/mnt/matylda4/xluner01/F5-TTS/audio_playground/experiments/cz/non_causal" \
# --lang "cs" \
# --verbose

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WER and CER for generated audio using Whisper transcriptions.")
    
    parser.add_argument(
        "--generated_audio_folder", 
        type=str, 
        required=True, 
        help="Path to the folder containing generated audio files."
    )
    parser.add_argument(
        "--reference_transcriptions_folder", 
        type=str, 
        required=True, 
        help="Path to the folder containing reference transcriptions."
    )
    parser.add_argument(
        "--output_folder_plots",
        type=str, 
        help="Folder to save WER/CER plots. If not provided, nothing is saved."
    )
    parser.add_argument(
        "--output_folder_results",
        type=str, 
        help="Folder to save text file with results. If not provided, nothing is saved."
    )
    parser.add_argument(
        "--plot_title", 
        type=str,
        default="TTS Evaluation", 
        help="Title of the WER/CER plot. Default: 'TTS Evaluation'."
    )
    parser.add_argument(
        "--lang", 
        required=True,
        default="en",
        choices=["en", "cs"],
        type=str,
        help="Language of the audio files."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="If set, print additional information during evaluation."
    )

    return parser.parse_args()

def clear_gpu_cache():
    """Clear the GPU cache."""
    with torch.no_grad():
      torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
    
def print_verbose(text):
    if args.verbose:
        print(text)
  
def load_whisper():
    """Load the Whisper model."""
    # clear_gpu_cache()
    print("Loading Whisper model...")
    global WHISPER_MODEL
    WHISPER_MODEL = whisper.load_model("large-v3", download_root=WHISPER_MODEL_DIR, device=DEVICE)
    print_verbose(f"Loaded Whisper model on {DEVICE} device.")
  
def transcribe_audio(audio_file):
    """Generate transcriptions for the given audio file."""
    # clear_gpu_cache()
    
    return WHISPER_MODEL.transcribe(audio_file, language=args.lang)
  
def process_input_files():
    """Process the input files."""
    # example: "experiment_default_per_sentence"
    experiment_names = [d for d in os.listdir(args.generated_audio_folder) if os.path.isdir(os.path.join(args.generated_audio_folder, d))]

    if args.output_folder_results is not None:
        output_file = os.path.join(args.output_folder_results, "wer_cer_results.txt")
        
        with open(output_file, "a") as f:
            f.write(f'\nSpeaker: {(args.generated_audio_folder).split("/")[-1]}\n')  # example: babis
    
    for experiment_name in experiment_names:
        WER_VALUES.clear()
        CER_VALUES.clear()
        
        print_verbose(f"WER and CER values should be [] []: {WER_VALUES} {CER_VALUES}")
        print(f"Processing experiment: {experiment_name}")
        generated_audio_files = os.listdir(os.path.join(args.generated_audio_folder, experiment_name))  # list all files
        generated_audio_files = [f for f in generated_audio_files if f.endswith(".wav")]                # filter only .wav files
        for file in tqdm(generated_audio_files):
            transcription_ref = os.path.join(args.reference_transcriptions_folder, file.replace(".wav", ".txt"))
            with open(transcription_ref, "r") as f:
                transcription_ref = f.read().strip()
            
            transcription_gen = transcribe_audio(os.path.join(args.generated_audio_folder, experiment_name, file))["text"]
            
            wer_and_cer(transcription_ref, transcription_gen)
        
        print(f"Mean WER: {np.mean(WER_VALUES):.2f}%")
        print(f"Mean CER: {np.mean(CER_VALUES):.2f}%")
        
        if args.output_folder_results is not None:
            with open(output_file, "a") as f:
                f.write(f'{experiment_name} {np.mean(WER_VALUES):.2f} {np.mean(CER_VALUES):.2f}\n')
            
        plot_and_save_results(WER_VALUES, CER_VALUES, args.plot_title, experiment_name)
    
def wer_and_cer(ground_truth, hypothesis):
  transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveKaldiNonWords(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
        ])

  # print information about WER & CER
  if args.verbose:
    print("====================================")
    h = hypothesis
    h = jiwer.ToLowerCase()(h)
    h = jiwer.ExpandCommonEnglishContractions()(h)
    h = jiwer.RemoveKaldiNonWords()(h)
    h = jiwer.RemoveWhiteSpace(replace_by_space=True)(h)
    h = jiwer.RemovePunctuation()(h)
    h = jiwer.RemoveMultipleSpaces()(h)
    h = jiwer.Strip()(h)

    g = ground_truth
    g = jiwer.ToLowerCase()(g)
    g = jiwer.ExpandCommonEnglishContractions()(g)
    g = jiwer.RemoveKaldiNonWords()(g)
    g = jiwer.RemoveWhiteSpace(replace_by_space=True)(g)
    g = jiwer.RemovePunctuation()(g)
    g = jiwer.RemoveMultipleSpaces()(g)
    g = jiwer.Strip()(g)

    print("H:", h)
    print("G:", g)

    print("WER: " + str(math.floor((wer(g, h, truth_transform=transformation, hypothesis_transform=transformation)*100))) + " %")
    print("CER: " + str(math.floor((cer(g, h, truth_transform=jiwer.cer_default, hypothesis_transform=jiwer.cer_default)*100))) + " %")

  WER_VALUES.append(wer(ground_truth, hypothesis, truth_transform=transformation, hypothesis_transform=transformation)*100)       # *100 because we want percentage
  CER_VALUES.append(cer(ground_truth, hypothesis, truth_transform=jiwer.cer_default, hypothesis_transform=jiwer.cer_default)*100) # *100 because we want percentage


def plot_and_save_results(wer_values, cer_values, title, experiment_name):
    if args.output_folder_plots is None or args.plot_title is None:
      return
    else:
      os.makedirs(args.output_folder_plots, exist_ok=True)
      
    """Plot and save boxplots of WER and CER values."""
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # WER Plot
    axes[0].boxplot(wer_values)
    axes[0].set_title(f"WER\n{title}")
    axes[0].set_ylabel("WER (%)")
    axes[0].set_xlabel(f"Mean WER: {np.mean(wer_values):.2f}%")

    # CER Plot
    axes[1].boxplot(cer_values)
    axes[1].set_title(f"CER\n{title}")
    axes[1].set_ylabel("CER (%)")
    axes[1].set_xlabel(f"Mean CER: {np.mean(cer_values):.2f}%")

    plt.savefig(os.path.join(args.output_folder_plots, f"wer_cer_boxplot_{experiment_name}.png"))
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    
    load_whisper()
    
    process_input_files()
