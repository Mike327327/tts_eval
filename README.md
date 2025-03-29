# tts_eval
TTS evaluation repository

## Installation
```bash
conda create -n tts_eval python=3.10
conda activate tts_eval
git clone https://github.com/Mike327327/tts_eval.git
cd tts_eval
pip install -r requirements.txt
```

## Usage
```bash
# To compute speaker similarity, run:
python speaker_similarity.py args_tbd

# To compute WER, run:
python wer_cer.py args_tbd
```