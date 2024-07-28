import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from transformer import Transformer


def create_masks(src, tgt, src_pad_idx, tgt_pad_idx):
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(
        2)  # Adjust to [batch_size, 1, 1, src_seq_len]
    tgt_mask = (tgt != tgt_pad_idx).unsqueeze(1).unsqueeze(
        2)  # Adjust to [batch_size, 1, 1, tgt_seq_len]
    tgt_mask = tgt_mask & (1 - torch.triu(torch.ones((1, tgt.size(-1),
                           tgt.size(-1)), device=tgt.device), diagonal=1)).bool()
    return src_mask, tgt_mask


# Example usage
src_vocab_size = 10000  # Adjust based on your data
tgt_vocab_size = 10000  # Adjust based on your data
src_pad_idx = 0  # Adjust based on your data
tgt_pad_idx = 0  # Adjust based on your data

model = Transformer(src_vocab_size, tgt_vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

for epoch in range(10):
    model.train()
    optimizer.zero_grad()

    src = torch.randint(0, src_vocab_size, (32, 10))  # Example source batch
    tgt = torch.randint(0, tgt_vocab_size, (32, 10))  # Example target batch
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    src_mask, tgt_mask = create_masks(src, tgt_input, src_pad_idx, tgt_pad_idx)
    preds = model(src, tgt_input, src_mask, tgt_mask)

    loss = criterion(preds.view(-1, preds.size(-1)), tgt_output.cont)
    print(loss)
