import torch
import polars as pl
import numpy as np
from pathlib import Path
from .stories_dataset import StoriesDataset

# inference
def inference(model, audio_embedded, text_embedded, modal_type):
    def _infer(infer_model, audio_inputs, text_inputs):
        match modal_type:
            case "MultiModal":
                return infer_model(audio_inputs, text_inputs)
            case "TextModal":
                return infer_model(text_inputs)
            case "AudioModal":
                return infer_model(audio_inputs)
        return None
    model.eval()
    with torch.no_grad():
        return _infer(model, audio_embedded, text_embedded)

if __name__ == "__main__":
    CKPT_PATH = Path("./ckpt/MultiModal-acc=0.650-e170-cls-1735382813.pth")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATASET_VOCAL_EMBED = Path("../datasets/vocal_embedded")
    DATASET_TEXT_EMBED = Path("../datasets/text_embedded")
    DATASET_ATTR_CSV = Path("../datasets/CBU0521DD_stories_attributes.csv")
    dataset_attr_df = pl.read_csv(DATASET_ATTR_CSV)

    dataset = StoriesDataset(dataset_attr_df, DATASET_TEXT_EMBED, DATASET_VOCAL_EMBED)
    model = torch.load(CKPT_PATH)
    modal_type = "MultiModal"

    rand_idx = np.random.randint(0, len(dataset)-1, 1).item()
    item = dataset.__getitem__(rand_idx)

    audio_embedded = torch.from_numpy(item["vocal"][np.newaxis,:]).to(DEVICE)
    text_embedded = torch.from_numpy(item["text"][np.newaxis,:]).to(DEVICE)

    result = inference(model, audio_embedded, text_embedded, modal_type)
    result = result.to(torch.device("cpu")).numpy().item()

    filename = dataset_attr_df.row(rand_idx, named=True)["filename"]
    result = "True Story" if result > 0.5 else "Deceptive Story"
    true_result = dataset_attr_df.row(rand_idx, named=True)["Story_type"]
    print(f"[{filename}]\n\tinfer result: {result}\n\t true result: {true_result}")