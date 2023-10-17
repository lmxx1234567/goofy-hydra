import torch.nn as nn
import torch.nn.functional as F
import torch

class ThroughputPredictor(nn.Module):
    def __init__(self,in_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, seq_len=100):
        super(ThroughputPredictor, self).__init__()
        self.linear_projection = nn.Linear(in_features, d_model)
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward
        )
        self.fc_out = nn.Linear(d_model, 1)  # Assuming the output is a single value per timestep

    def forward(self, src, tgt=None):
        src = self.linear_projection(src)  # (seq_len, batch_size, d_model)
        # If tgt is None, use the last value of src as the start of the tgt sequence
        if tgt is None:
            tgt = src[-1:, :, :]  # Assume the last value of src is the start of the tgt sequence
        else:
            tgt = self.linear_projection(tgt)

        transformer_output = self.transformer(src, tgt) # (seq_len, batch_size, d_model)

        # Use max pooling to convert the transformer output to a single vector per sequence
        pooled_output = torch.max(transformer_output, dim=0)[0]  # (batch_size, d_model)

        output = F.relu(self.fc_out(pooled_output), inplace=True)

        return output
    
class LinkSelector(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, num_links):
        super(LinkSelector, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward),
            num_layers=num_encoder_layers
        )
        self.fc_out = nn.Linear(d_model, num_links)  # Output a score for each link
        
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