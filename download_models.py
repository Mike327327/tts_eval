# !git clone https://github.com/speechbrain/speechbrain.git
# %cd /content/speechbrain
# !pip install -r requirements.txt
# !pip install --editable .
# import speechbrain as sb
# from speechbrain.pretrained import EncoderClassifier
# classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb") # speechbrain/spkrec-xvect-voxceleb, savedir="pretrained_models/spkrec-xvect-voxceleb"
# %cd /content

# !pip install git+https://github.com/openai/whisper.git
# !pip install jiwer

# import whisper
# import jiwer
# from jiwer import wer, cer
# try:
#     import tensorflow  # required in Colab to avoid protobuf compatibility issues
# except ImportError:
#     pass

# def clear_gpu_cache():
#     with torch.no_grad():
#       torch.cuda.empty_cache()
#     torch.cuda.empty_cache()
#     gc.collect()
#     model = None

# model = whisper.load_model("large-v2")