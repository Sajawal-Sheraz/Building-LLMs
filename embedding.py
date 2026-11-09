import os
import torch
import tiktoken
from dataset import create_dataloader

print("--- 1. Reading Text from File ---")
file_path = "data/the-verdict.txt"

# Check if the file exists, if not provide a friendly reminder
if not os.path.exists(file_path):
    print(f"Error: Could not find {file_path}. Please make sure the 'data' folder and 'the-verdict.txt' file exist!")
else:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    print(f"Successfully loaded file! Total characters: {len(raw_text)}")

    print("\n--- 2. Creating DataLoader via Tiktoken ---")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create the dataloader using our sliding window from dataset.py
    dataloader = create_dataloader(
        txt=raw_text, 
        tokenizer=tokenizer, 
        batch_size=2, 
        max_length=4, 
        stride=4, 
        shuffle=False
    )

    # Pull the first batch of inputs and targets
    for batch_inputs, batch_targets in dataloader:
        print("Batch Inputs shape:", batch_inputs.shape)
        print("Batch Targets shape:", batch_targets.shape)
        print("First Batch Input IDs:", batch_inputs[0])
        print("First Batch Target IDs:", batch_targets[0])
        
        # Save one batch input to pass into the embedding layers next
        sample_input_ids = batch_inputs[0]
        break

    print("\n--- 3. Passing Batch Token IDs into Embedding Layers ---")
    vocab_size = 50257   # Standard GPT-2 vocabulary size
    output_dim = 256     # Embedding vector dimension
    context_length = len(sample_input_ids)

    # Define embedding layers
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    # Generate token embeddings and positional embeddings
    token_embeddings = token_embedding_layer(sample_input_ids)
    pos_ids = torch.arange(len(sample_input_ids))
    pos_embeddings = pos_embedding_layer(pos_ids)

    # Combine them (Token Embeddings + Positional Embeddings)
    input_embeddings = token_embeddings + pos_embeddings
    print("Final Input Embeddings Shape for this Batch:", input_embeddings.shape)
    print("Output Vector per Token:", f"({input_embeddings.shape[0]} tokens, {input_embeddings.shape[1]} dimensions)")