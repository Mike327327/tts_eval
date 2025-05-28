import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import glob
import librosa

# python extract_long_audio_files_parczech.py \
# --input_folder "/mnt/matylda4/xluner01/ParCzech/parczech-3.0-asr-train-2021" \
# --output_folder "./matches" \
# --keyword "Babis" \
# --min_duration_sec 8 \
# --max_duration_sec 20

def parse_args():
    parser = argparse.ArgumentParser(
        description="Find and copy .speakers files containing a keyword and whose corresponding audio duration falls within a given range."
    )
    parser.add_argument(
        "--input_folder",
        type=Path,
        required=True,
        help="Root folder to search (recursively) for .speakers files.",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Folder to copy matched files into.",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        required=True,
        help="Keyword to search for inside .speakers files.",
    )
    parser.add_argument(
        "--min_duration_sec",
        type=float,
        default=0.0,
        help="Minimum allowed audio duration in seconds.",
    )
    parser.add_argument(
        "--max_duration_sec",
        type=float,
        default=float("inf"),
        help="Maximum allowed audio duration in seconds.",
    )
    return parser.parse_args()

def get_audio_duration(wav_path: Path) -> float:
    """Return the duration of the audio file in seconds, or 0 on error."""
    try:
        return librosa.get_duration(path=str(wav_path))
    except Exception as e:
        print(f"[!] Error reading duration for {wav_path}: {e}")
        return 0.0

def find_and_copy(root: Path, out: Path, keyword: str, min_dur: float, max_dur: float):
    out.mkdir(parents=True, exist_ok=True)
    count = 0

    pattern = str(root / "**" / "*.speakers")
    all_speakers = list(glob.glob(pattern, recursive=True))

    for spk_path_str in tqdm(all_speakers, desc="Scanning", unit="file"):
        spk_path = Path(spk_path_str)
        try:
            content = spk_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[!] Could not read {spk_path}: {e}")
            continue

        if keyword not in content:
            continue

        wav_path = spk_path.with_suffix(".wav")
        prt_path = spk_path.with_suffix(".prt")
        if not wav_path.exists() or not prt_path.exists():
            print(f"[!] Missing file for {spk_path.stem}:",
                  f"{'no .wav' if not wav_path.exists() else ''}",
                  f"{'no .prt' if not prt_path.exists() else ''}")
            continue

        duration = get_audio_duration(wav_path)
        if not (min_dur <= duration <= max_dur):
            continue

        # copy .speakers, .wav, .prt to flat output folder with unique names
        base = spk_path.stem.replace(".", "_")
        for ext in [".speakers", ".wav", ".prt"]:
            src = spk_path.with_suffix(ext)
            dst = out / f"{base}{ext}"
            shutil.copy2(src, dst)
        count += 1

    print(f"\n✅ Copied {count} sets of (.speakers, .wav, .prt) to {out}")

def main():
    args = parse_args()
    print(f"Searching in: {args.input_folder}")
    print(f"Keyword: '{args.keyword}'")
    print(f"Duration between {args.min_duration_sec}s and {args.max_duration_sec}s\n")
    find_and_copy(
        args.input_folder,
        args.output_folder,
        args.keyword,
        args.min_duration_sec,
        args.max_duration_sec
    )

if __name__ == "__main__":
    main()
