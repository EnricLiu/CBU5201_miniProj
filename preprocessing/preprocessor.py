import os
import gc
from pathlib import Path

import torch.cuda
from tqdm import tqdm
import polars as pl

from .loudness_norm import loudness_norm
from .msst_denoise import msst_denoise
from .voice2text import VTT, VttModelType

class Preprocessor:
    RENAMED_DIR     = "renamed"
    NORMALIZED_DIR  = "normalized"
    DENOISED_DIR    = "denoised"
    VOCAL_OUT_DIR   = "vocal"
    TEXT_OUT_DIR    = "text"

    def __init__(self, config: dict, tmp_path:Path=Path("./tmp")):
        self._tmp_path = tmp_path
        if self._tmp_path.is_file():
            raise FileExistsError("Temporary directory already exists")
        self._tmp_path.mkdir(parents=True, exist_ok=True)
        self._configs = config

        self._vtt_model = None # lazy loaded

    def get_denoise_config(self) -> dict:
        ret = self._configs.get("denoiseConfig")
        if ret is None:
            raise ValueError("No denoiseConfig found in config.json")
        return ret

    def vtt(self, audio: Path, audio_attr: pl.DataFrame, output: Path):
        def transcribe(audio_file: Path, out: Path):
            row = audio_attr.filter(pl.col("filename") == audio_file.name)
            srt_path = out.joinpath(audio_file.stem + ".srt")
            try:
                audio_lang = "zh" if row.get_column("Language").item() == "Chinese" else "en"
            except Exception:
                print(f"[WARN] Preprocessor: Failed to get language from {str(audio_file)}")
                audio_lang = "en"

            self._vtt_model.transcribe_to_srt(audio_file, srt_path, audio_lang)
            return audio_file

        if not audio.exists(): raise FileNotFoundError(f"{str(audio)} does not exist")
        print("-----------------Transcribing------------------")
        if self._vtt_model is None:
            print("[INFO] Preprocessor: Loading vtt model ...")
            self._vtt_model = VTT(VttModelType.LARGE)
            print("[INFO] Preprocessor: Finish loading!")

        output.mkdir(parents=True, exist_ok=True)
        if audio.is_file():
            return transcribe(audio, output)

        for audio in tqdm(list(audio.iterdir())):
            if audio.is_file(): transcribe(audio, output)

        return audio

    def process(self, audio: Path, attr_df: pl.DataFrame, out_path: Path, stage: str="all", max_workers: int=os.cpu_count()):
        if out_path.is_file(): raise NotADirectoryError(f"{str(out_path)} is not a directory")
        out_path.mkdir(parents=True, exist_ok=True)
        vocal_out_path = out_path / self.VOCAL_OUT_DIR
        text_out_path =  out_path / self.TEXT_OUT_DIR
        stage = {"all":0, "norm":1, "denoise":2, "denoise_norm":3, "vtt":4}[stage]
        tasks = [
            lambda a: loudness_norm(a, self._tmp_path / self.NORMALIZED_DIR, ac=2, max_workers=max_workers),
            lambda a: msst_denoise(a, self._tmp_path / self.DENOISED_DIR, self.get_denoise_config()),
            lambda a: loudness_norm(a, vocal_out_path, max_workers=max_workers),
            lambda a: self.vtt(a, attr_df, text_out_path)
        ]

        for task in tasks[stage:]: audio = task(audio)

        print(f"[INFO] Preprocessor: process succeed.")
        print(f"[INFO] Preprocessor: vocal@{vocal_out_path.absolute()}, text@{text_out_path.absolute()}")
        return audio, vocal_out_path, text_out_path

    def cleanup(self):
        if input("y" != "Are you sure you want to delete all files in the temporary directory: {}? [y/N]"
                            .format(self._tmp_path.absolute())): return False

        for path in self._tmp_path.iterdir():
            if path.is_dir():
                os.rmdir(path)
            else:
                path.unlink()

        return True

    def destruct(self):
        del self._vtt_model
        gc.collect()
        torch.cuda.empty_cache()



if __name__ == "__main__":
    import json
    configs = json.load(open("./config.json"))
    # audio_attr = pl.read_csv("../datasets/dataset_attr.csv")
    # row = audio_attr.filter(pl.col("filename") == "00001.wav")
    # print(row)
    # audio_lang = "zh" if row.get_column("Language").item() == "Chinese" else "en"
    # print(audio_lang)

    preprocessor = Preprocessor(configs)
    preprocessor.process(audio=Path("./test/00001.wav"), attr_csv=Path("../datasets/dataset_attr.csv"), out_path=Path("./out"))