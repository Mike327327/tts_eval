import argparse
import os
import re
from pathlib import Path
import soundfile as sf

# Example usage:
# python ./dataset_processing/prepare_libritts_eval.py \
#     --input_folder "/home/m/F5-TTS/audio_playground/test-clean/LibriTTS/test-clean" \
#     --output_folder "/home/m/F5-TTS/audio_playground/en/to_generate" \
#     --max_duration_sec 12 \
#     --target_minutes 10

def parse_args():
    parser = argparse.ArgumentParser(
        description="Obtain transcriptions used for evaluation tests of the model trained on ParCzech."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing the extracted parczech-3.0-asr-context files."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder to export the filtered transcription txt files."
    )
    parser.add_argument(
        "--max_duration_sec",
        type=float,
        default=8,
        help="Maximum allowed recording duration in seconds (files longer than this are filtered out)."
    )
    parser.add_argument(
        "--target_minutes",
        type=float,
        default=10,
        help="Target total speech duration (in minutes) for the selection (approximately)."
    )
    return parser.parse_args()

def get_audio_duration(audio_path):
    """
    Return the duration of the audio file (in seconds).
    """
    try:
        info = sf.info(str(audio_path))
        return info.duration
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return 0

def main():
    args = parse_args()
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 1. Obtain all .normalized.txt file paths (recursively)
    transcriptions = list(input_folder.rglob("*.normalized.txt"))
    print(f"Found {len(transcriptions)} .normalized.txt files in {input_folder}")
    
    valid_files_dict = {}
    # 2. Filter out those with numerical or special characters in the file name (without extension)
    # Here, we filter out if the file stem contains any digit or any of the following: #, &, etc.
    for transcription in transcriptions:
        file_stem = transcription.stem
        # read the file stem and check if it contains any of the following characters
        with open(transcription, "r", encoding="utf-8") as f:
          text = f.readline().strip()
        if re.search(r"[0-9#&§@%&\*\(\)_+\-=\[\]\{\};:'\"]", text):
          continue
        
        # 3. Assume the corresponding audio file is the same name with a .wav extension
        transcription = transcription.with_name(transcription.stem)  # Removes '.txt' part
        transcription = transcription.with_name(transcription.stem)  # Removes '.normalized' part
        
        audio_file = transcription.with_suffix(".wav")
        if not audio_file.exists():
            continue
        
        duration_sec = get_audio_duration(audio_file)
        duration_min = duration_sec / 60  # convert seconds to minutes
        
        valid_files_dict[file_stem] = [str(transcription.with_suffix("")), duration_min]
    
    print(f"After filtering by file name, {len(valid_files_dict)}/{len(transcriptions)} files remain.")

    # 4. Filter out recordings that are above the max_duration_sec limit
    max_duration_min = args.max_duration_sec / 60
    filtered_files = {name: info for name, info in valid_files_dict.items() if info[1] <= max_duration_min}
    print(f"After filtering recordings above {args.max_duration_sec} sec, {len(filtered_files)}/{len(transcriptions)} files remain.")

    # 5. Accumulate files until a target total duration (in minutes) is reached.
    selected_files = {}
    total_duration = 0.0
    # Sorting files by duration (ascending) so that shorter recordings are added first
    for name, info in filtered_files.items():
        if total_duration >= args.target_minutes:
            break
        selected_files[name] = info
        total_duration += info[1]
    print(f"Selected {len(selected_files)} files with total duration of {total_duration:.2f} minutes (target was {args.target_minutes} minutes).")
    
    # # 6. Export the transcription from each selected .prt file to an individual .txt file in the output folder.
    # for name, info in selected_files.items():
    #     # The base path for the .prt file (without extension)
    #     prt_base_path = Path(info[0])
    #     prt_path = prt_base_path.with_suffix(".normalized.txt")
    #     try:
    #         with open(prt_path, "r", encoding="utf-8") as f:
    #             transcription = f.read().strip()
    #         # Save the transcription in a txt file named after the file stem
    #         output_txt_path = output_folder / f'{name.split(".")[0]}.txt'
    #         with open(output_txt_path, "w", encoding="utf-8") as out_f:
    #             out_f.write(transcription)
    #     except Exception as e:
    #         print(f"Error processing {prt_path}: {e}")
            
    import shutil  # Make sure this is at the top with other imports

    # 6.1 Copy corresponding .wav files into the output folder
    for name, info in selected_files.items():
        prt_base_path = Path(info[0])
        wav_path = prt_base_path.with_suffix(".wav")
        output_wav_path = output_folder / f'{name.split(".")[0]}.wav'
        print(f"Copying {wav_path} to {output_wav_path}")

        try:
            if wav_path.exists():
                shutil.copy2(wav_path, output_wav_path)
            else:
                print(f"Warning: WAV file not found for {name}")
        except Exception as e:
            print(f"Error copying WAV file {wav_path}: {e}")

    
    # 7. Print final statistics
    print("\n=== Statistics ===")
    print(f"Total .prt files found: {len(transcriptions)}")
    print(f"Valid files after name filtering: {len(valid_files_dict)}")
    print(f"Files after duration filtering: {len(filtered_files)}")
    print(f"Files selected for export: {len(selected_files)}")
    print(f"Total duration of selected files: {total_duration:.2f} minutes")
    
if __name__ == "__main__":
    main()
