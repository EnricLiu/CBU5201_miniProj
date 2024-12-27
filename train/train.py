import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader, random_split

from model import CrossModalTransformerModel, TextModalTransformerModel
from stories_dataset import StoriesDataset

def train_model(model, device, dataset_df: pd.DataFrame, text_path: Path, vocal_path: Path, ckpt_path: Path | None, val_percent: float = 0.1, amp=False, epochs=10, learning_rate=1e-4, batch_size=1, modal_type: str = "MultiModal", model_out_method: str = "cls"):
    # ret vals definitions
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    cplt_epoches = 0

    dataset = StoriesDataset(dataset_df, text_path, vocal_path)
    # print(dataset.__getitem__(1))

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    match modal_type:
        case "MultiModal":
            model_summary = summary(model, input_size=[(1, 10240, 512), (1, 512, 1536)])
        case "TextModal":
            model_summary = summary(model, input_size=(1, 512, 1536))
        case _:
            raise NotImplementedError
    model_params = model_summary.to_megabytes(model_summary.total_input + model_summary.total_output_bytes + model_summary.total_param_bytes)

    print(f'''Starting training:
        Epochs:          {epochs}
        Model size:      {model_params} MBytes
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {ckpt_path.absolute()}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    def infer(infer_model, audio_inputs, text_inputs):
        match modal_type:
            case "MultiModal":
                return infer_model(audio_inputs, text_inputs)
            case "TextModal":
                return infer_model(text_inputs)
            case "AudioModal":
                return infer_model(audio_inputs)
        return None

    try:
        for epoch in range(1, epochs+1):
            model.train()
            epoch_loss = 0.0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}') as pbar:
                for batch in train_loader:
                    audio_embedded = batch["vocal"].to(device)
                    text_embedded = batch["text"].to(device)
                    labels = batch["label"].to(device)

                    # 前向传播
                    outputs = infer(model, audio_embedded, text_embedded)
                    # print(f"output: {outputs.squeeze()}\t label: {labels}")
                    loss = criterion(outputs.squeeze(), labels.squeeze().float())

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.update(batch_size)
                    epoch_loss += loss.item()
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                epoch_loss /= len(train_loader)
                # print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.3f}")

            # train acc
            model.eval()
            train_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in train_loader:
                    audio_embedded = batch["vocal"].to(device)
                    text_embedded = batch["text"].to(device)
                    labels = batch["label"].to(device)  # 假设标签在数据集中

                    outputs = infer(model, audio_embedded, text_embedded)
                    loss = criterion(outputs.squeeze(), labels.squeeze().float())
                    train_loss += loss.item()

                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            train_epoch_loss = train_loss / len(train_loader)
            train_accuracy = correct / total
            print(f"[Train] Loss: {train_epoch_loss:.4f}, Acc: {train_accuracy:.4f}", end="\t")

            # 验证
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            # temp = []
            with torch.no_grad():
                for batch in val_loader:
                    audio_embedded = batch["vocal"].to(device)
                    text_embedded = batch["text"].to(device)
                    labels = batch["label"].to(device)  # 假设标签在数据集中

                    outputs = infer(model, audio_embedded, text_embedded)
                    loss = criterion(outputs.squeeze(), labels.squeeze().float())
                    val_loss += loss.item()

                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    # temp.append((outputs, labels))

            val_epoch_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            print(f"[Validation] Loss: {val_epoch_loss:.4f}, Acc: {val_accuracy:.4f}", end="\t")
            # print(temp)

            if ckpt_path and epoch % 5 == 0:
                ckpt_path.mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                # state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, os.path.join(ckpt_path, f'{modal_type}-acc={val_accuracy:.3f}-e{epoch}-{model_out_method}-{datetime.now().strftime("%Y%m%d_%H_%M_%S")}.pth'))
                print(f'Checkpoint {epoch} saved!')
            else:
                print()

            train_accs.append(train_accuracy)
            train_losses.append(train_epoch_loss)
            val_accs.append(val_accuracy)
            val_losses.append(val_epoch_loss)
            cplt_epoches = epoch

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        if ckpt_path:
            print("Saving last checkpoint...")
            ckpt_path.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, os.path.join(ckpt_path, f'{modal_type}-AUTOSAVE-e{cplt_epoches[-1]}-{model_out_method}-{datetime.now().strftime("%Y%m%d_%H_%M_%S")}.pth'))
            print(f'Checkpoint {epoch} saved!')
    finally:
        return train_losses, train_accs, val_losses, val_accs, cplt_epoches

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CKPT_PATH = Path("./ckpt")

    DATASET_VOCAL_EMBED = Path("../datasets/vocal_embedded")
    DATASET_TEXT_EMBED = Path("../datasets/text_embedded")
    DATASET_ATTR_CSV = Path("../datasets/dataset_attr.csv")
    dataset_attr_df = pd.read_csv(DATASET_ATTR_CSV)

    audio_dim = 512
    text_dim = 1536

    # Text-modal
    # d_model = 512
    # nhead = 16
    # num_layers = 12
    # dim_feedforward = 8192
    # dropout = 0.1
    # model_out_method = "cls"
    # modal_type = "TextModal"
    # text_model = TextModalTransformerModel(text_dim, d_model=text_dim, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, output_method=model_out_method, dropout=0.1)
    # train_loss, train_acc, val_loss, val_acc, epoches \
    #     = train_model(
    #         text_model,
    #         DEVICE,
    #         dataset_df  = dataset_attr_df,
    #         text_path   = DATASET_TEXT_EMBED,
    #         vocal_path  = DATASET_VOCAL_EMBED,
    #         ckpt_path   = CKPT_PATH,
    #         amp =   False,
    #         val_percent     = 0.1,
    #         learning_rate   = 1e-7,
    #         epochs          = 100,
    #         batch_size      = 1,
    #         modal_type      = "TextModal",
    #         model_out_method = "cls"
    #     )

    # Multi-modal
    d_model = 512
    nhead = 8
    num_layers = 6
    dim_feedforward = 3072
    dropout = 0.1
    model_out_method = "cls"
    modal_type = "MultiModal"
    model = CrossModalTransformerModel(audio_dim, text_dim, d_model, nhead, num_layers, dim_feedforward, dropout).to(
        DEVICE)
    train_loss, train_acc, val_loss, val_acc, epoches \
        = train_model(
        model,
        DEVICE,
        dataset_df=dataset_attr_df,
        text_path=DATASET_TEXT_EMBED,
        vocal_path=DATASET_VOCAL_EMBED,
        ckpt_path=CKPT_PATH,
        amp=False,
        val_percent=0.1,
        learning_rate=3e-6,
        epochs=100,
        batch_size=1,
        modal_type=modal_type,
        model_out_method=model_out_method,
    )

    import matplotlib.pyplot as plot

    plot.figure(figsize=(24, 10))
    ax = plot.subplot(1, 2, 1)
    ax.plot(train_loss, label="train loss")
    ax.plot(val_loss, label="val loss")
    ax.set_title(f"Loss over Epochs({modal_type} {model_out_method})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax = plot.subplot(1, 2, 2)
    ax.plot(train_acc, label="train acc")
    ax.plot(val_acc, label="val acc")
    ax.set_title(f"Accuracy over Epochs({modal_type} {model_out_method})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plot.savefig(Path(f"./results/{modal_type}-{model_out_method}-{datetime.now().strftime("%Y%m%d_%H_%M_%S")}.png"),
                 bbox_inches='tight')
    plot.show()
