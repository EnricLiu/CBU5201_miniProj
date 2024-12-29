import math
import torch
import torch.nn as nn

from .encoding import PositionalEncoding, TransformerEncoder

class TextModalTransformerModel(nn.Module):
    def __init__(self, text_dim: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float, max_len: int=10000, output_method: str = 'cls', num_types: int=2):
        super(TextModalTransformerModel, self).__init__()

        self.text_dim = text_dim
        self.d_model = d_model
        self.output_method = output_method

        # 1. 维度转换层
        self.text_linear = nn.Linear(text_dim, d_model)

        # 2. Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # 4. Cross-Modal Transformer Encoder
        self.transformer_encoder = TransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)

        self.average_pooling = nn.AdaptiveAvgPool1d(1)
        # 5. 分类头 (这里简化为一个线性层 + sigmoid)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, text_embeddings):
        embedding = self.text_linear(text_embeddings)   # (batch_size, text_seq_len, text_dim) -> (batch_size, text_seq_len, d_model)
        batch_size, seq_len, _ = embedding.shape

        match self.output_method:
            case 'cls':
                cls_t = torch.zeros(batch_size, 1, self.d_model, device=embedding.device)  # [CLS_T] token
                embedding = torch.cat([cls_t, embedding], dim=1)
            case 'avg_pool':
                pass
            case _:
                pass

        embedding = self.positional_encoding(embedding * math.sqrt(self.d_model)) # 通常需要对embedding进行scale
        embedding = self.transformer_encoder(embedding)

        match self.output_method:
            case 'cls':
                output = embedding[:, 0, :]
            case 'avg_pool':
                output = self.average_pooling(embedding.transpose(1, 2)).squeeze(2)
            case _:
                output = None

        output = self.classifier(output)
        return output