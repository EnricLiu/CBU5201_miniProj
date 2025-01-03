from pathlib import Path
import subprocess
import shutil
import os

def _read_config(msst_config:dict):
    msst_root = Path(__file__).resolve().parent / msst_config["msstRoot"]
    return {
        "python":       msst_root / msst_config["workenv"] / "python.exe",
        ".py":          msst_root / msst_config["infer_py"],
        "config":       msst_root / msst_config["toolConfig"],
        "model_ckpt":   msst_root / msst_config["modelCkpt"],
        "model_type":   msst_config["modelType"],
    }

def msst_denoise(input: Path, output: Path, msst_config: dict):
    if not input.exists(): raise FileNotFoundError(f"Input dir {input} not found")
    configs = _read_config(msst_config)
    output.mkdir(parents=True, exist_ok=True)

    input_folder = input
    if input.is_file():
        tmp_folder = output.parent.joinpath("_denoise_tmp")
        tmp_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(input, tmp_folder / input.name)
        input_folder = tmp_folder

    print("------------------MSST-Denoise------------------")
    # print(configs)
    res = subprocess.run([
        configs["python"], "-u", configs[".py"],
        "--config_path",        str(configs["config"]),
        "--start_check_point",  str(configs["model_ckpt"]),
        "--input_folder",       str(input_folder),
        "--store_dir",          str(output),
        "--model_type",         str(configs["model_type"]),
        "--device_ids",         "0",
    ], stderr=subprocess.PIPE)

    if res.returncode != 0:
        print(res.stderr.decode("utf-8"))
        raise RuntimeError("MSST-Denoise failed")

    for audio in list(output.iterdir()):
        if not audio.is_dir(): continue
        shutil.move(audio / "vocals.wav", output / (audio.stem + ".wav"))

    if input.is_file():
        shutil.rmtree(input_folder)
        return output.joinpath(input.name)

    return output

if __name__ == "__main__":
    import json
    config = json.load(open("./config.json"))["denoiseConfig"]
    # Path("./tmp/denoised/test_vocals.wav").rename(Path("./tmp/denoised/test.wav"))
    msst_denoise(Path("./test"), Path("./denoised_result"), config)
    # msst_denoise(Path("./test/test.wav"), Path("./test/denoised_result"), config)