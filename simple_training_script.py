import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "This train goes fast",
    "Its snowing today",
    "The football game was okey"
]
batch = tokenizer(sequences, padding=True,
                  truncation=True, return_tensors="pt")

# # This is new
#batch["labels"] = torch.tensor([1, 1])

# optimizer = AdamW(model.parameters())
# loss = model(**batch).loss
# loss.backward()
# optimizer.step()
