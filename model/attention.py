import torch
import torch.nn as nn


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # Trainable weight matrices for Queries, Keys, and Values
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        # x shape: (batch_size, num_tokens, d_in)

        # 1. Project inputs into Queries, Keys, and Values
        queries = self.W_query(x)  # Shape: (b, num_tokens, d_out)
        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values = self.W_value(x)  # Shape: (b, num_tokens, d_out)

        # 2. Compute attention scores (Dot Product of Q and transposed K)
        # keys.transpose(1, 2) flips the last two dimensions for matrix multiplication
        attn_scores = queries @ keys.transpose(
            1, 2
        )  # Shape: (b, num_tokens, num_tokens)

        # 3. Scale scores by the square root of the dimension size and apply Softmax
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / (d_k**0.5), dim=-1)

        # 4. Multiply by Values to get the final context vectors
        context_vec = attn_weights @ values  # Shape: (b, num_tokens, d_out)

        return context_vec


# Test script to verify SelfAttention_v1
if __name__ == "__main__":
    # Let's say we have a batch of 2 sentences, 3 tokens each, and each token is a 4-dimensional vector
    torch.manual_seed(123)
    batch_size = 2
    num_tokens = 3
    d_in = 4
    d_out = 2

    # Dummy input tensor (e.g., our token embeddings + positional embeddings)
    sample_inputs = torch.randn(batch_size, num_tokens, d_in)

    # Initialize our self-attention module
    attention_layer = SelfAttention_v1(d_in=d_in, d_out=d_out)

    # Run the forward pass
    context_vectors = attention_layer(sample_inputs)

    print("Input Tensor Shape:", sample_inputs.shape)
    print("Output Context Vectors Shape:", context_vectors.shape)
    print("\nContext Vectors:\n", context_vectors)
