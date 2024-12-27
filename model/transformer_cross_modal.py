import math
import torch
import torch.nn as nn

from encoding import TypeEncoding, PositionalEncoding, TransformerEncoder


class CrossModalTransformerModel(nn.Module):
    def __init__(self, audio_dim, text_dim, d_model, nhead, num_layers, dim_feedforward, dropout, max_len=10000, num_types=2):
        super(CrossModalTransformerModel, self).__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.d_model = d_model

        # 1. 维度转换层
        self.audio_linear = nn.Linear(audio_dim, d_model)
        self.text_linear = nn.Linear(text_dim, d_model)

        # 2. Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # 3. Type Encoding
        self.type_encoding = TypeEncoding(d_model, num_types)

        # 4. Cross-Modal Transformer Encoder
        self.cross_modal_encoder = TransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)

        # 5. 分类头 (这里简化为一个线性层 + sigmoid)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, audio_embeddings, text_embeddings):
        # 1. 准备输入序列
        audio_embeddings = self.audio_linear(audio_embeddings) # (batch_size, audio_seq_len, audio_dim) -> (batch_size, audio_seq_len, d_model)
        text_embeddings = self.text_linear(text_embeddings)   # (batch_size, text_seq_len, text_dim) -> (batch_size, text_seq_len, d_model)

        batch_size, audio_seq_len, _ = audio_embeddings.shape
        _, text_seq_len, _ = text_embeddings.shape

        cls_a = torch.zeros(batch_size, 1, self.d_model, device=audio_embeddings.device)  # [CLS_A] token
        cls_t = torch.zeros(batch_size, 1, self.d_model, device=audio_embeddings.device)  # [CLS_T] token
        sep = torch.zeros(batch_size, 1, self.d_model, device=audio_embeddings.device)    # [SEP] token

        combined_embeddings = torch.cat([cls_a, audio_embeddings, sep, cls_t, text_embeddings], dim=1) # 拼接

        # 3. 添加位置编码和类型编码
        type_ids = torch.cat([
            torch.zeros(batch_size, audio_seq_len + 2, dtype=torch.long, device=audio_embeddings.device), # 0 for audio, [CLS_A], [SEP]
            torch.ones(batch_size, text_seq_len + 1, dtype=torch.long, device=audio_embeddings.device)   # 1 for text, [CLS_T]
        ], dim=1)
        combined_embeddings = self.positional_encoding(combined_embeddings * math.sqrt(self.d_model)) # 通常需要对embedding进行scale
        combined_embeddings = self.type_encoding(combined_embeddings, type_ids)

        # 4. Cross-Modal Transformer Encoder
        encoded_embeddings = self.cross_modal_encoder(combined_embeddings)

        # 5. 分类 (这里取 [CLS_A] token 的输出, 你也可以尝试其他方式)
        cls_a_output = encoded_embeddings[:, 0, :]
        cls_t_output = encoded_embeddings[:, audio_seq_len + 2, :]
        cls_output = cls_a_output + cls_t_output # 可以将cls_a 和 cls_t的输出相加, 或者拼接
        output = self.classifier(cls_output)

        return output