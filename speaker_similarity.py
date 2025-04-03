import torchaudio
import numpy as np
import speechbrain as sb
from speechbrain.inference import EncoderClassifier
import argparse
import os
from tqdm import tqdm

CLASSIFIER = None
SPEECHBRAIN_MODEL_DIR = "./pretrained_models/speechbrain-spkrec"
COSINE_DISTANCE_VALUES = []

# python speaker_similarity.py \
# --generated_audio_folder "/mnt/matylda4/xluner01/F5-TTS/audio_playground/experiments/cz/non_causal/babis" \
# --reference_audio_file "/mnt/matylda4/xluner01/F5-TTS/audio_playground/cz/reference/ref_audio_cz_babis.wav" \
# --output_folder_results "/mnt/matylda4/xluner01/F5-TTS/audio_playground/experiments/cz/non_causal" \
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
      "--reference_audio_file", 
      type=str, 
      required=True, 
      help="Path to the folder containing reference audio file."
  )
  parser.add_argument(
      "--output_folder_results",
      type=str, 
      help="Folder to save text file with results. If not provided, nothing is saved."
  )
  parser.add_argument(
      "--verbose", 
      action="store_true", 
      help="If set, print additional information during evaluation."
  )

  return parser.parse_args()

def print_verbose(text):
  if args.verbose:
    print(text)

def load_classifier():
  """Load the Speechbrain model."""
  print_verbose("Loading Speechbrain model for embedding extraction...")
  global CLASSIFIER
  CLASSIFIER = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=SPEECHBRAIN_MODEL_DIR)
  
def compute_embedding(wav_path):
  signal, fs = torchaudio.load(wav_path)
  signal = torchaudio.functional.resample(signal, orig_freq=fs, new_freq=16000)
  speechbrain_embedding = CLASSIFIER.encode_batch(signal)
  embedding = speechbrain_embedding.numpy()[0,0]
  return embedding

# computes cosine distance in order to pre-filter the wavs list
def compute_cosine_distance(ref_wav_file, gen_wav_file):
  if not os.path.exists(ref_wav_file):
    raise FileNotFoundError(f"Reference audio file not found: {ref_wav_file}")
  if not os.path.exists(gen_wav_file):
    raise FileNotFoundError(f"Generated audio file not found: {gen_wav_file}")
  
  ref_embedding = compute_embedding(ref_wav_file)
  gen_embedding = compute_embedding(gen_wav_file)

  # calculate the cosine distance between the model and converted wav
  model_norm = (ref_embedding.T / np.linalg.norm(ref_embedding, axis=0))
  embed_norm = (gen_embedding.T / np.linalg.norm(gen_embedding, axis=0)).T
  cos_distance = 1-embed_norm @ model_norm

  return cos_distance

def process_input_files():
  """Process the input files."""
  experiment_names = [d for d in os.listdir(args.generated_audio_folder) if os.path.isdir(os.path.join(args.generated_audio_folder, d))]

  if args.output_folder_results is not None:
    output_file = os.path.join(args.output_folder_results, "speaker_similarity_results.txt")
    
    with open(output_file, "a") as f:
        f.write(f'\nSpeaker: {(args.generated_audio_folder).split("/")[-1]}\n')  # example: babis
            
  for experiment_name in experiment_names:
    print(f"Processing experiment: {experiment_name}")
    
    COSINE_DISTANCE_VALUES.clear()
    print_verbose(f"Cosine distance values should be []: {COSINE_DISTANCE_VALUES}")
    
    generated_audio_files = os.listdir(os.path.join(args.generated_audio_folder, experiment_name))  # list all files
    generated_audio_files = [f for f in generated_audio_files if f.endswith(".wav")]                # filter only .wav files
    
    for gen_file in tqdm(generated_audio_files):
      cos_distance = compute_cosine_distance(args.reference_audio_file, os.path.join(args.generated_audio_folder, experiment_name, gen_file))
      COSINE_DISTANCE_VALUES.append(cos_distance)
      print_verbose(f"Cosine distance: {cos_distance:.2f}, file: {gen_file}")

    print(f"Mean cosine distance: {np.mean(COSINE_DISTANCE_VALUES):.2f} for experiment: {experiment_name}")
    
    if args.output_folder is not None:
      with open(output_file, "a") as f:
        f.write(f'{experiment_name} {np.mean(COSINE_DISTANCE_VALUES):.2f}\n')
      
if __name__ == "__main__":
  args = parse_args()
  
  load_classifier()
  
  process_input_files()