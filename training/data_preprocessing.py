import torch
import os

def preprocess_corpus(filename):
    if os.path.isfile(filename):

        with open(filename, 'r') as obj:
            lyrics = obj.read()

        chars = sorted(list(set(lyrics)))
        str2int = {v: k for k, v in enumerate(chars)}
        int2str = {k: v for k, v in enumerate(chars)}
        encode = lambda line: [str2int[s] for s in line]
        decode = lambda line: ''.join([int2str[s] for s in line])
        data = encode(lyrics)

        # Create train, dev, and test splits
        n1 = int(0.9 * len(data))
        n2 = int(0.8 * len(data))
        train_data = data[:n1]
        dev_data = data[n2:n1]
        test_data = data[n1:]
        data = {'train': train_data, 'dev': dev_data, 'test': test_data}

    return (data, encode, decode, chars)


def get_batches(split, batch_size=64, context_len=256, device='cpu'):

    ix = torch.randint(len(split) - context_len, (batch_size,), device=device)  # Ensure index tensor is on GPU
    x = torch.stack([torch.tensor(split[i:i + context_len], device=device) for i in ix])
    y = torch.stack([torch.tensor(split[i + 1:i + context_len + 1], device=device) for i in ix])
    return x, y