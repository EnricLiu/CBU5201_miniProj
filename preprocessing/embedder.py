import re
import gc
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from wav_embed import wav_tknzr_embed
from voice2text import read_srt

class Embedder:
    VOCAL_EMBED_DIR   = "vocal_embedded"
    TEXT_EMBED_DIR    = "text_embedded"

    def __init__(self, config: dict, tmp_path:Path=Path("./tmp")):
        self._configs = config

        self._tmp_path = tmp_path
        if self._tmp_path.is_file(): raise FileExistsError("Temporary directory already exists")
        self._tmp_path.mkdir(parents=True, exist_ok=True)

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._qwen_tokenizer = None # lazy loaded
        self._qwen_model = None     # lazy loaded

    def get_wav_tknzr_config(self) -> dict:
        ret = self._configs.get("wavTokenizerConfig")
        if ret is None:
            raise ValueError("No wavTokenizerConfig found in config.json")
        return ret

    def text_embed(self, srt: Path, attr_df: pl.DataFrame, output: Path):

        def get_embed(sentences: list[str], lang: str = "zh"):
            def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
                left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
                if left_padding:
                    return last_hidden_states[:, -1]
                else:
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_size = last_hidden_states.shape[0]
                    return last_hidden_states[
                        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

            text = ""
            punctuation_dict = {"zh": "，", "en": ","}
            for sentence in sentences:
                sentence = sentence.strip()
                text += sentence
                if not is_punctuation(sentence):
                    text += punctuation_dict[lang]
            inputs = self._qwen_tokenizer(text, max_length=2048, padding=True, truncation=True, return_tensors="pt").to(
                self.DEVICE)
            with torch.no_grad():
                outputs = self._qwen_model(**inputs)
                _pool = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                _pool = F.normalize(_pool, p=2, dim=1)
                _embedding = F.normalize(outputs.last_hidden_state, p=2, dim=1)
                # scores = (embeddings[:2] @ embeddings[2:].T) * 100
            return _embedding, _pool, inputs["attention_mask"]

        def qwen_embed(srt_file: Path, out: Path):
            row = attr_df.filter(pl.col("filename") == (srt.stem + ".wav"))
            try:
                lang = "zh" if row.get_column("Language").item() == "Chinese" else "en"
            except Exception:
                print(f"[WARN] Preprocessor: Failed to get language from {str(srt_file)}")
                lang = "en"

            sentences = read_srt(srt_file)
            embedding, pool, _ = get_embed(sentences, lang)
            save_path = out.joinpath(srt.stem + ".npz")
            np.savez(save_path, embedding=embedding.cpu().numpy(), pool=pool.cpu().numpy())
            return save_path

        if not self._qwen_model or not self._qwen_tokenizer:
            print("Loading Qwen Model...")
            self._qwen_tokenizer = AutoTokenizer.from_pretrained(
                "Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
            self._qwen_model = AutoModel.from_pretrained(
                "Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True).to(self.DEVICE)
            print("Qwen Model Loaded")

        output = output / self.TEXT_EMBED_DIR
        output.mkdir(parents=True, exist_ok=True)
        print("-------------------Text-Embed-------------------")
        if srt.is_file():
            return qwen_embed(srt, output)

        for srt in tqdm(list(srt.iterdir())):
            qwen_embed(srt, output)

        return output

    def vocal_embed(self, vocal: Path, output: Path):
        output = output / self.VOCAL_EMBED_DIR
        return wav_tknzr_embed(vocal, output, self.get_wav_tknzr_config())

    def destruct(self):
        del self._qwen_model
        del self._qwen_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

def is_punctuation(s: str):
    pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~！“”‘’（）【】《》〈〉；：，。？、]$'
    return bool(re.search(pattern, s))


if __name__ == "__main__":
    import json
    config = json.load(open("./config.json"))
    embedder = Embedder(config)
    df = pl.read_csv("../datasets/dataset_attr.csv")
    embedder.text_embed(Path("./out/text"), df, Path("./out"))
    embedder.vocal_embed(Path("./out/vocal"), Path("./out"))

