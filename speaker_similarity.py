import torchaudio
import numpy as np
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
import argparse
import os
from tqdm import tqdm

CLASSIFIER = None
SPEECHBRAIN_MODEL_DIR = "./pretrained_models/speechbrain-spkrec"
COSINE_DISTANCE_VALUES = []

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
      "--verbose", 
      action="store_true", 
      help="If set, print additional information during evaluation."
  )

  return parser.parse_args()

def load_classifier():
  """Load the Speechbrain model."""
  global CLASSIFIER
  CLASSIFIER = EncoderClassifier.from_hparams(source=SPEECHBRAIN_MODEL_DIR, savedir=SPEECHBRAIN_MODEL_DIR)
  
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
  gen_audio_files = os.listdir(args.generated_audio_folder)

  for gen_file in tqdm(gen_audio_files):
    ref_file = gen_file.replace("_gen.wav", ".wav")
    cos_distance = compute_cosine_distance(os.path.join(args.reference_audio_folder, ref_file), os.path.join(args.generated_audio_folder, gen_file))
    COSINE_DISTANCE_VALUES.append(cos_distance)
    if args.verbose:
      print(f"Cosine distance: {cos_distance:.2f}, file: {gen_file}")

  print(f"Mean cosine distance: {np.mean(COSINE_DISTANCE_VALUES):.2f}")
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Evaluate speaker similarity TTS transcriptions.")
  args = parser.parse_args()
  
  load_classifier()
  
  process_input_files()