import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ):
        super(TrafficScheduler, self).__init__()
        self.state_feature_extractor = SharedStateFeatureExtractor.get_instance(
            in_features,
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
        )

        self.fc0 = nn.Linear(d_model, d_model // 2)
        self.fc1 = nn.Linear(d_model // 2, 3)

    def forward(self, src, tgt=None):
        if tgt is None:
            tgt = src

        features = self.state_feature_extractor(src)  # (batch_size, seq_len, d_model)

        fc_out = self.fc0(features)
        output = self.fc1(fc_out)

        output = torch.softmax(output, dim=-1) # (batch_size, seq_len, 3)
        return output


class StateValueCritic(nn.Module):
    def __init__(
        self,
        in_features,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
    ):
        super(StateValueCritic, self).__init__()
        self.state_feature_extractor = SharedStateFeatureExtractor.get_instance(
            in_features,
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
        )

        self.fc0 = nn.Linear(d_model, d_model // 2)
        self.fc1 = nn.Linear(d_model // 2, 1)

    def forward(self, src, tgt=None):
        if tgt is None:
            tgt = src

        features = self.state_feature_extractor(src)  # (batch_size, seq_len, d_model)

        # Use max pooling to convert the transformer output to a single vector per sequence
        seq_len = features.shape[1]
        if torch.is_tensor(seq_len): # for tracing
            seq_len = seq_len.item()
        pooled_output = F.max_pool1d(
            features.transpose(1, 2), seq_len
        ).squeeze(
            -1
        )  # (batch_size, d_model)

        fc_out = self.fc0(pooled_output)  # (batch_size, d_model//2)
        output = self.fc1(fc_out)  # (batch_size, 1)

        output = F.relu(output, inplace=True)  # (batch_size, 1)
        return output


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trafficScheduler = TrafficScheduler(19, 256, 8, 3, 3, 2048).to(device)
    src = torch.zeros(1, 100, 19).to(device)
    output = trafficScheduler(src)
    print(output.shape)

    # save model
    import os

    if not os.path.exists("saved_models/trafficScheduler"):
        os.makedirs("saved_models/trafficScheduler")

    torch.save(
        trafficScheduler.state_dict(), "saved_models/trafficScheduler/untrained.pt"
    )

    # save traced model
    traced_script_module = torch.jit.trace(trafficScheduler, src)
    traced_script_module.save("saved_models/trafficScheduler/untrain_traced.pt")
