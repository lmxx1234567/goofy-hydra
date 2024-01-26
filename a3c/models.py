import torch
import torch.nn as nn
import torch.nn.functional as F
""" for transformer """
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TransformerModel(nn.Module):
    def __init__(
            self, 
            in_features: int = 19, 
            d_model: int = 512, 
            nhead: int = 2, 
            dim_feedforward: int = 200,
            num_encoder_layers: int = 2, 
            dropout: float = 0.5
            ):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.embedding = nn.Linear(in_features, d_model) # in_features=19, d_model=512
        self.d_model = d_model
        self.linear = nn.Linear(d_model, in_features)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, in_features]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[0]).to(src.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output
    
    _instance = None
    @classmethod
    def get_instance(
        cls,
        in_features: int = 19, 
        d_model: int = 512, 
        nhead: int = 2, 
        dim_feedforward: int = 200,
        num_encoder_layers: int = 2, 
        dropout: float = 0.5
    ):
        if cls._instance is None:
            cls._instance = cls(
                in_features, 
                d_model, 
                nhead, 
                dim_feedforward,
                num_encoder_layers, 
                dropout
            )
        return cls._instance

    def set_grad_requires(self, requires_grad):
        for param in self.linear_projection.parameters():
            param.requires_grad = requires_grad
    
class SharedStateFeatureExtractor(nn.Module):  # SSFE
    def __init__(
        self,
        in_features,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
    ):
        super(SharedStateFeatureExtractor, self).__init__()
        self.linear_projection = nn.Linear(in_features, d_model)

        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            batch_first=True,
        )

    def forward(self, src, tgt=None):
        src = self.linear_projection(src)  # (batch_size, seq_len, in_features)
        if tgt is None:
            tgt = src
        else:
            tgt = self.linear_projection(tgt)

        output = self.transformer(src, tgt)  # (batch_size, seq_len, d_model)
        return output

    _instance = None

    @classmethod
    def get_instance(
        cls,
        in_features,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
    ):
        if cls._instance is None:
            cls._instance = cls(
                in_features,
                d_model,
                nhead,
                num_encoder_layers,
                num_decoder_layers,
                dim_feedforward,
            )
        return cls._instance

    def set_grad_requires(self, requires_grad):
        for param in self.linear_projection.parameters():
            param.requires_grad = requires_grad


class TrafficScheduler(nn.Module):
    def __init__(
        self,
        in_features,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        seq_len=100,
        dropout = 0.2,
    ):
        super(TrafficScheduler, self).__init__()
        self.state_feature_extractor = TransformerModel.get_instance(
            in_features, 
            d_model, 
            nhead, 
            dim_feedforward,
            num_encoder_layers, 
            dropout,
        )
        self.seq_len = seq_len

        self.scheduler = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Linear(in_features // 2, 2)
        )
        self.value = nn.Sequential(
            nn.Linear(in_features * seq_len, in_features * seq_len // 2),
            nn.Linear(in_features * seq_len // 2, 1)
        )

    def forward(self, src):
        trans_out = self.state_feature_extractor(src)
        policy_output = self.scheduler(trans_out)
        policy_output = F.relu(policy_output)  # (batch_size, 1)
        policy_output = torch.softmax(policy_output, dim=-1)  # (batch_size, seq_len, 2)
        # policy_output = torch.clip(policy_output, 0.40, 0.90)

        value_src = trans_out.reshape(
            -1, self.seq_len * trans_out.shape[-1]
        )  # (batch_size, seq_len* in_features)
        value_output = self.value(value_src)
        value_output = F.relu(value_output)  # (batch_size, 1)
        return policy_output, value_output


class StateValueCritic(nn.Module):
    def __init__(
        self,
        in_features,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        seq_len=100,
        dropout = 0.2,
    ):
        super(StateValueCritic, self).__init__()
        self.state_feature_extractor = TransformerModel.get_instance(
            in_features, 
            d_model, 
            nhead, 
            dim_feedforward,
            num_encoder_layers, 
            dropout
        )

        self.seq_len = seq_len

        self.fc0 = nn.Linear(in_features * seq_len, in_features * seq_len // 2)
        self.fc1 = nn.Linear(in_features * seq_len // 2, 1)

    def forward(self, src):
        src = self.state_feature_extractor(src)
        src = src.reshape(
            -1, self.seq_len * src.shape[-1]
        )  # (batch_size, seq_len* in_features)
        fc_out = self.fc0(src)  # (batch_size, seq_len * in_features // 2)
        output = self.fc1(fc_out)  # (batch_size, 1)

        output = F.relu(output, inplace=True)  # (batch_size, 1)
        return output


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trafficScheduler = TrafficScheduler(19, 256, 8, 3, 3, 2048).to(device)
    src = torch.zeros(1, 100, 19).to(device)
    output = trafficScheduler(src)
    print(output)
    print(output.shape)

    stateValueCritic = StateValueCritic(19, 256, 8, 3, 3, 2048).to(device)
    output = stateValueCritic(src)
    print(output)
    print(output.shape)
