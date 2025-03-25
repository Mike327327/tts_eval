import whisper
from speechbrain.pretrained import EncoderClassifier
import os

# Directory to save the models locally
SAVE_DIR = "./pretrained_models"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_whisper_model(model_name="large-v2"):
    """Download and save the Whisper model locally."""
    print(f"Downloading Whisper model: {model_name}")
    model = whisper.load_model(model_name, download_root=SAVE_DIR)
    print(f"Whisper model '{model_name}' saved locally at {SAVE_DIR}/whisper.")

def download_speechbrain_model():
    """Download and save the SpeechBrain speaker recognition model locally."""
    print("Downloading SpeechBrain speaker recognition model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=f"{SAVE_DIR}/speechbrain-spkrec",
    )
    print(f"SpeechBrain model saved locally at {SAVE_DIR}/speechbrain-spkrec.")

if __name__ == "__main__":
    download_whisper_model("large-v2")
    download_speechbrain_model()
    print("All models downloaded and saved locally.")
