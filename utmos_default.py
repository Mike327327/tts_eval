import utmosv2

print("Init")
model = utmosv2.create_model(pretrained=True)

print("Model loaded")

mos = model.predict(input_path="/mnt/matylda4/xluner01/F5-TTS/audio_playground/en/reference/ref_audio_en_david_jaquay.wav")
print(mos)

mos = model.predict(input_path="/mnt/matylda4/xluner01/F5-TTS/audio_playground/en/reference/ref_audio_en_david_jaquay.wav")
print(mos)

mos = model.predict(input_path="/mnt/matylda4/xluner01/F5-TTS/audio_playground/en/reference/ref_audio_en_david_jaquay.wav")
print(mos)

