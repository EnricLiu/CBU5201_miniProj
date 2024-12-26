import argparse
from pathlib import Path
from encoder.utils import convert_audio
import torchaudio
import torch
import numpy as np
from tqdm import tqdm
from decoder.pretrained import WavTokenizer

def infer(tokenizer, audio: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(audio)
    wav = convert_audio(wav, sr, 24000, 1)
    bandwidth_id = torch.tensor([0])
    wav = wav.to(device)
    features, discrete_code = tokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
    return features

# def recursive_infer_save(tokenizer, curr_path: Path, out_path: Path):
#     for nxt_path in curr_path.iterdir():
#         if nxt_path.is_dir():
#             out_nxt_path = out_path.joinpath(nxt_path.name)
#             out_nxt_path.mkdir(exist_ok=True)
#             recursive_infer_save(nxt_path, out_nxt_path)
#         if nxt_path.suffix == ".wav":
#             result = infer(tokenizer, nxt_path).cpu().numpy()
#             np.save(out_path.joinpath(f"{nxt_path.name}.npy"), result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--start_check_point", type=str)
    parser.add_argument("--store_dir", type=str)
    parser.add_argument("--input_folder", type=str)

    args = parser.parse_args()
    config  = args.config_path
    model   = args.start_check_point
    inputs  = args.input_folder
    outputs = args.store_dir

    if config is None or not Path(config).is_file():
        raise ValueError(f"Invalid config path {config}.")
    if model is None or not Path(model).is_file():
        raise ValueError(f"Invalid model path {model}.")
    if inputs is None or not Path(inputs).is_dir():
        raise ValueError(f"Invalid input folder {inputs}.")
    if outputs is None:
        raise ValueError(f"Invalid output folder {outputs}.")

    config = Path(config)
    model = Path(model)
    inputs = Path(inputs)
    outputs = Path(outputs)
    outputs.mkdir(parents=True, exist_ok=True)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tknzr = WavTokenizer.from_pretrained0802(config, model).to(device)

    # # chunking to reduce CPU-GPU mem exchange
    # CHUNK_SIZE = 64
    # audios = [p for p in inputs.iterdir() if p.suffix == ".wav"]
    # pbar = tqdm(range(len(audios)))
    # audios_chunks = [audios[i:i+CHUNK_SIZE] for i in range(0, len(audios), CHUNK_SIZE)]
    # for audios in audios_chunks:
    #     results = []
    #     for audio in audios:
    #         results.append((audio.name, infer(tknzr, audio)))
    #         pbar.update(1)
    #
    #     results = map(lambda x: (x[0], x[1].cpu().numpy()), results)
    #     for name, result in results:
    #         np.save(outputs.joinpath(f"{name.rstrip(".wav")}.npy"), result)

    for audio_path in tqdm(list(inputs.iterdir())):
        result = infer(tknzr, audio_path).to("cpu").numpy()
        save_path = outputs.joinpath(audio_path.stem + ".npy")
        np.save(save_path, result)
