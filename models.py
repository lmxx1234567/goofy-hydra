import torch.nn as nn
import torch.nn.functional as F
import torch


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
        self._is_pretraining = False
        self.linear_projection = nn.Linear(in_features, d_model)

        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            batch_first=True,
        )

        self.fc = nn.Linear(d_model, in_features)  # fc for pretraining

    def forward(self, src, tgt, tgt_mask=None, tgt_is_causal=None):
        src = self.linear_projection(src)  # (batch_size, seq_len, in_features)
        tgt = self.linear_projection(tgt)  # (batch_size, seq_len, in_features)
        tgt_mask, tgt_is_causal = None, None

        output = self.transformer(
            src, tgt, tgt_mask=tgt_mask, tgt_is_causal=tgt_is_causal
        )  # (batch_size, seq_len, d_model)

        # Use liner to convert the transformer output to tgt shape for pretraining
        if self._is_pretraining:
            output = self.fc(output)  # (batch_size, seq_len, in_features)
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
        for p in self.parameters():
            p.requires_grad = requires_grad

    def set_pretraining(self, is_pretraining: bool):
        self._is_pretraining = is_pretraining

    def init_weights(self, init_fn):
        """
        Initialize model weights.
        :param init_fn: Function to use for weight initialization.
        """
        for p in self.parameters():
            if p.dim() > 1:
                init_fn(p)


class ThroughputPredictor(nn.Module):
    def __init__(
        self,
        in_features,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
    ):
        super(ThroughputPredictor, self).__init__()
        self.state_feature_extractor = SharedStateFeatureExtractor.get_instance(
            in_features,
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
        )
        self.fc0 = nn.Linear(d_model, d_model // 2)
        # Assuming the output is a single value per timestep
        self.fc_out = nn.Linear(d_model // 2, 1)

    def forward(self, src, tgt=None):
        transformer_output = self.state_feature_extractor(
            src, tgt
        )  # (batch_size, seq_len, d_model)

        # Use max pooling to convert the transformer output to a single vector per sequence
        pooled_output = F.max_pool1d(
            transformer_output.transpose(1, 2), transformer_output.shape[1]
        ).squeeze(
            -1
        )  # (batch_size, d_model)

        output = self.fc0(pooled_output)
        output = self.fc_out(output)

        return output


class ThroughputPredictorLiner(nn.Module):
    def __init__(self, in_features, d_model, seq_len=100):
        super(ThroughputPredictorLiner, self).__init__()
        self.linear_projection = nn.Linear(in_features, d_model)
        self.fc0 = nn.Linear(d_model, d_model // 2)
        # Assuming the output is a single value per timestep
        self.fc_out = nn.Linear(d_model // 2, 1)

    def forward(self, src, tgt=None):
        src = self.linear_projection(src)  # (batch_size, seq_len, d_model)

        # Use max pooling to convert the transformer output to a single vector per sequence
        pooled_output = F.max_pool1d(src.transpose(1, 2), src.shape[1]).squeeze(
            -1
        )  # (batch_size, d_model)

        output = self.fc0(pooled_output)
        output = self.fc_out(output)

        return output


class LinkSelector(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, num_links):
        super(LinkSelector, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward),
            num_layers=num_encoder_layers,
        )
        # Output a score for each link
        self.fc_out = nn.Linear(d_model, num_links)

    def forward(self, src):
        # src shape: (1, link_num, 6)
        src = src.permute(1, 0, 2)  # Change shape to (link_num, 1, 6)
        transformer_output = self.transformer_encoder(src)
        # Sum over the transformer output across the link dimension
        # to obtain a single vector representation
        sum_output = transformer_output.sum(dim=0)
        link_scores = self.fc_out(sum_output)
        # Apply softmax to get probabilities, and reshape to 1xlink_num
        link_probs = F.softmax(link_scores, dim=-1).unsqueeze(0)
        return link_probs


if __name__ == "__main__":
    # Test ThroughputPredictorLiner
    model = ThroughputPredictor(8, 256, 4, 3, 3, 2048)
    src = torch.randn(1, 100, 6)
    output = model(src)
    print(output.shape)
