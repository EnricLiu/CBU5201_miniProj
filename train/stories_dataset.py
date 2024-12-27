from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class StoriesDataset(Dataset):
    def __init__(self, dataset_df: pd.DataFrame, text_path: Path, vocal_path: Path):
        self.df = dataset_df
        self.text_path = text_path
        self.vocal_path = vocal_path

    def __getitem__(self, idx):
        if idx >= len(self):
            return None
        filename_stem = ".".join(self.df.loc[idx, 'filename'].split(".")[:-1])
        vocal_embedded_path = self.vocal_path.joinpath(filename_stem + ".npy")
        text_embedded_path = self.text_path.joinpath(filename_stem + ".npz")
        # (bs, seq_len, dim)
        vocal_embedded = np.load(vocal_embedded_path)
        text_embedded = np.load(text_embedded_path)["embedding"][0, :, :]
        vocal_embedded = np.transpose(vocal_embedded, (0, 2, 1))[0, :, :]
        label = np.asarray([[1 if self.df.loc[idx, 'label'] else 0]], dtype=np.float32)
        return { "text": text_embedded, "vocal": vocal_embedded, "label": label }

    def __len__(self):
        return len(self.df)
