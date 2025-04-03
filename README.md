# tts_eval
TTS evaluation repository

## Installation
```bash
conda create -n tts_eval python=3.10
conda activate tts_eval
git clone https://github.com/Mike327327/tts_eval.git
cd tts_eval
pip install -r requirements.txt

### Whisper (ASR)
git+https://github.com/openai/whisper.git

### Speechbrain (embedding extractor)
git+https://github.com/speechbrain/speechbrain.git

# git clone https://github.com/speechbrain/speechbrain.git
# cd /content/speechbrain
# pip install -r requirements.txt
```

## Usage
```bash
# To compute speaker similarity, run:
# (run it first without GPU, the model gets downloaded, then cached for later)
python speaker_similarity.py \
--generated_audio_folder "/path/to/generated/audio/F5-TTS/audio_playground/experiments/cz/non_causal/babis" \
--reference_audio_file "/path/to/ref/texts/F5-TTS/audio_playground/cz/ref_seen_speaker.wav" \
--output_folder_results "/path/for/outputting/results/txtfile/F5-TTS/audio_playground/experiments/cz/non_causal" \
--verbose \

# To compute WER, run:
# (run it first without GPU, the model gets downloaded, then cached for later)
python wer_cer.py \
--generated_audio_folder "/path/to/generated/audio/F5-TTS/audio_playground/experiments/cz/non_causal/babis" \
--reference_transcriptions_folder "/path/to/ref/texts/F5-TTS/audio_playground/cz/to_generate" \
--output_folder_plots "path/for/outputting/plot/images/F5-TTS/audio_playground/experiments/cz/non_causal/babis" \
--plot_title "Czech non-causal unseen male speaker" \
--output_folder_results "/path/for/outputting/results/txtfile/F5-TTS/audio_playground/experiments/cz/non_causal" \
--lang "cs" \
--verbose \
```