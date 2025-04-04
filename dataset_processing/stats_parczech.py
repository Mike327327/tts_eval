import os
from tqdm import tqdm

# Folder path with .prt files
folder_path = "/mnt/matylda4/xluner01/ParCzech/parczech-3.0-asr-context.test"

# Collect all .prt files
prt_files = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".prt"):
            prt_files.append(os.path.join(root, file))

print(f"Found {len(prt_files)} .prt files.\n")

# Loop through the files with tqdm and count characters
total_chars = 0
for file_path in tqdm(prt_files, desc="Processing .prt files"):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        total_chars += len(content)

print(f"\nTotal characters in all .prt files: {total_chars}")
print(f"Average characters per .prt file: {total_chars / len(prt_files):.2f}")

# import os
# from tqdm import tqdm

# folder_path = "/mnt/matylda4/xluner01/ParCzech/parczech-3.0-asr-train-2020"

# # Step 1: Find all .speakers files
# speaker_files = []
# for root, dirs, files in os.walk(folder_path):
#     for file in files:
#         if file.endswith(".speakers"):
#             speaker_files.append(os.path.join(root, file))

# print(f"Found {len(speaker_files)} .speakers files.\n")

# # Step 2: Read speaker names
# all_speakers = set()

# for file_path in tqdm(speaker_files, desc="Processing .speakers files"):
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             speaker = line.strip()
#             if speaker:  # skip empty lines
#                 all_speakers.add(speaker)

# # Step 3: Output result
# print(f"\nTotal unique speakers: {len(all_speakers)} in {folder_path}.")