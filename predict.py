import torch
from models import ThroughputPredictor

if __name__ == "__main__":
    # Create a ThroughputPredictor instance
    # Note: the values for the parameters are arbitrary and are only used for demonstration purposes
    #       in this example
    predictor = ThroughputPredictor(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
    )

    # Create a random input tensor
    # Note: the values for the input tensor are arbitrary and are only used for demonstration purposes
    #       in this example
    src = torch.rand(10, 32, 512)
    tgt = torch.rand(20, 32, 512)

    # Run the model
    output = predictor(src, tgt)
    print(output.shape)