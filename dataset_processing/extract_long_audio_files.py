import argparse
import shutil
from pathlib import Path
import librosa

# Usage:
# python extract_long_audio_files.py \
# --input_folder "/home/m/datasets/petrpavel_interview134_clean" \
# --output_folder "/home/m/datasets/petrpavel_interview134_clean_long" \
# --min_duration_sec 10 \
# --max_duration_sec 20

# python extract_long_audio_files.py \
# --input_folder "/home/m/datasets/danusenerudova_interview34_clean" \
# --output_folder "/home/m/datasets/danusenerudova_interview34_clean_long" \
# --min_duration_sec 10 \
# --max_duration_sec 20

def parse_args():
    parser = argparse.ArgumentParser(description="Extract .wav and .prt files longer than a threshold duration.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the source dataset root folder.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save the filtered files.")
    parser.add_argument("--min_duration_sec", type=float, default=10.0, help="Minimum duration (in seconds) to keep files.")
    parser.add_argument("--max_duration_sec", type=float, default=20.0, help="Maximum duration (in seconds) to keep files.")
    return parser.parse_args()

def get_audio_duration(path):
    try:
        return librosa.get_duration(path=str(path))
    except Exception as e:
        print(f"Error reading duration for {path}: {e}")
        return 0

def flatten_filename(path: Path, root: Path) -> str:
    relative_parts = path.relative_to(root).with_suffix('').parts
    return "_".join(relative_parts)

def collect_and_copy_files(input_folder, output_folder, min_duration, max_duration):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    count = 0

    for wav_path in input_folder.rglob("*.wav"):
        duration = get_audio_duration(wav_path)
        if duration >= min_duration and duration < max_duration:
            prt_path = wav_path.with_suffix(".prt")
            if prt_path.exists():
                flat_name = flatten_filename(wav_path, input_folder)
                output_wav_path = output_folder / f"{flat_name}.wav"
                output_prt_path = output_folder / f"{flat_name}.prt"

                shutil.copy2(wav_path, output_wav_path)
                shutil.copy2(prt_path, output_prt_path)
                count += 1
            else:
                print(f"Missing .prt for: {wav_path}")

    print(f"\n✅ Copied {count} .wav+.prt pairs longer than {min_duration} seconds to {output_folder}")

def main():
    args = parse_args()
    collect_and_copy_files(args.input_folder, args.output_folder, args.min_duration_sec, args.max_duration_sec)

if __name__ == "__main__":
    main()
