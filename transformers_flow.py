from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
import torch
#Like other neural networks, Transformer models can’t process raw text directly, so the first step of our pipeline is to convert the text inputs into numbers that the model can make sense of. To do this we use a tokenizer, which will be responsible for:
#Splitting the input into words, subwords, or symbols(like punctuation) that are called tokens
#Mapping each token to an integer
#Adding additional inputs that may be useful to the model
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "The movie was really good",
    "Team lost the match",
]
inputs = tokenizer(raw_inputs, padding=True,
                   truncation=True, return_tensors="pt")
print('tokenized inputs',inputs)

# For each model input, we’ll retrieve a high-dimensional vector representing the contextual 
# understanding of that input by the Transformer model.
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

# Since this is a sequence classification task, AutoModelForSequenceClassification
# is better suited.

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

# logits are the raw, unnormalized scores outputted by the last layer of the model
print('AutoModelForSequenceClassification logits', outputs.logits)
print('AutoModelForSequenceClassification logits shape' , outputs.logits.shape)

print(model.config.id2label)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)


