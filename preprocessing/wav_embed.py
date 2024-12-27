from pathlib import Path
import subprocess
import shutil

def _read_config(tknzr_config:dict):
    tknzr_root = Path(Path.cwd() / tknzr_config["tknzrRoot"])
    return {
        "python":       tknzr_root / tknzr_config["workenv"] / "python.exe",
        ".py":          tknzr_root / tknzr_config["infer_py"],
        "config":       tknzr_root / tknzr_config["toolConfig"],
        "model_ckpt":   tknzr_root / tknzr_config["modelCkpt"],
    }

def wav_tknzr_embed(input: Path, output: Path, wav_tknzr_config: dict):
    if not input.exists(): raise FileNotFoundError(f"Input dir {input} not found")
    configs = _read_config(wav_tknzr_config)
    output.mkdir(parents=True, exist_ok=True)

    input_folder = input
    if input.is_file():
        tmp_folder = output.parent.joinpath("_wav_embed_tmp")
        tmp_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(input, tmp_folder / input.name)
        input_folder = tmp_folder

    print("-------------------VocalEmbed-------------------")
    subprocess.run([
        configs["python"], configs[".py"],
        "--config_path",        str(configs["config"]),
        "--start_check_point",  str(configs["model_ckpt"]),
        "--input_folder",       str(input_folder),
        "--store_dir",          str(output),
    ])

    if input.is_file():
        shutil.rmtree(input_folder)
        return output.joinpath(input.name)

    return output