import os
import ffmpeg
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def loudness_norm(audios: Path, output: Path,
                  out_fmt: str="wav", ac: int=1, ar: int=48000, max_workers: int=os.cpu_count()):
    def norm(audio: Path):
        if audio.is_dir(): return None
        output.mkdir(parents=True, exist_ok=True)
        out_path = output.joinpath(audio.name)
        try:
            (
                ffmpeg
                .input(str(audio), f=audio.suffix.lstrip("."))
                .filter("loudnorm")
                .output(str(out_path), **{"ac": ac, "ar": ar}, f=out_fmt)
                .overwrite_output()
                .run()
            )
        except Exception as e:
            print(f"Error: {e} on audio {str(audio)}")

        return out_path

    if not audios.exists(): raise FileNotFoundError(f"{str(audios)} does not exist")
    print("------------------LoudnessNorm------------------")
    output.mkdir(parents=True, exist_ok=True)
    if audios.is_file(): return norm(audios)

    audios = list(audios.iterdir())
    pbar = tqdm(total=len(audios))
    _norm = lambda audio: norm(audio); pbar.update(1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        _results = executor.map(_norm, audios)

    return output
