import matplotlib.pyplot as plot
import seaborn as sns
import polars as pl
import numpy

import torch

from pathlib import Path

from .train import get_dataset

def draw_confusion_matrix(pred: numpy.ndarray, true: numpy.ndarray, title='Confusion matrix'):
    sns.heatmap(numpy.array([[numpy.sum((pred == i) & (true == j)) for j in range(2)] for i in range(2)]), annot=True, fmt='d', cmap="viridis")
    plot.title(title)
    plot.xlabel("Ground Truth")
    plot.ylabel("Predicted")
    plot.show()

if __name__ == "__main__":
    DATASET_VOCAL_EMBED = Path("../datasets/vocal_embedded")
    DATASET_TEXT_EMBED = Path("../datasets/text_embedded")
    DATASET_ATTR_CSV = Path("../datasets/CBU0521DD_stories_attributes.csv")
    dataset_attr_df = pl.read_csv(DATASET_ATTR_CSV)

    plot.figure(figsize=(10, 10))
    model = torch.load("./ckpt/MultiModal-acc=0.650-e170-cls-1735382813.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_dataset = get_dataset(dataset_attr_df, DATASET_TEXT_EMBED, DATASET_VOCAL_EMBED, 0, 1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model.eval()
    pred_result = []
    label_result = []
    with torch.no_grad():
        for batch in val_loader:
            audio_embedded = batch["vocal"].to(device)
            text_embedded = batch["text"].to(device)
            labels = batch["label"].to(device)

            outputs = model(audio_embedded, text_embedded)
            predicted = (outputs > 0.5).float()

            pred_result.append(predicted.cpu().squeeze().numpy())
            label_result.append(labels.cpu().squeeze().numpy())

    draw_confusion_matrix(numpy.asarray(pred_result), numpy.asarray(label_result))
    print("Accuracy: ", numpy.mean(numpy.asarray(pred_result) == numpy.asarray(label_result)))
    print(pred_result)
    print(label_result)
    plot.show()