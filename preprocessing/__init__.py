import polars as pl
from pathlib import Path
from loudness_norm import loudness_norm
from preprocessor import Preprocessor
from embedder import Embedder

def preprocess(input: Path, output: Path, attr_df: pl.DataFrame, config: dict, tmp_path: Path):
    print("-------------------Preprocess-------------------")
    p = Preprocessor(config, tmp_path)
    _, vocal_out, text_out = p.process(input, attr_df, output, stage="all")
    p.destruct()
    del p
    print("-------------------Embedding--------------------")
    e = Embedder(config, tmp_path)
    e.vocal_embed(vocal_out, output)
    e.text_embed(text_out, attr_df, output)
    e.destruct()

    return

if __name__ == "__main__":
    import json
    config = json.load(open("./config.json"))
    attr_df = pl.read_csv("../datasets/dataset_attr.csv")
    preprocess(Path("./test"), Path("./out"), attr_df, config, Path("./tmp"))