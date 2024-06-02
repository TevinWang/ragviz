import torch

def embedding_function(tokenizer, model, query):
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    decoder_input_ids = tokenizer(query, return_tensors="pt").input_ids

    # Forward pass through the model to obtain embeddings
    with torch.no_grad():
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

    # Extract the embeddings
    embeddings = outputs.last_hidden_state  # Last layer hidden states

    embeddings_np = embeddings.numpy()
    return embeddings_np[0,0].tolist()