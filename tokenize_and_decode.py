from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "New York is the most populous city in the United States"
# Tokenize the sentance
tokens = tokenizer.tokenize(sequence)

print(tokens)

# convert to input IDs
ids = tokenizer.convert_tokens_to_ids(tokens)

# These outputs, once converted to the appropriate framework tensor, can then be used as inputs to a model
print(ids)

decoded_string = tokenizer.decode(
    [1203, 1365, 1110, 1103, 1211, 22608, 1331, 1107, 1103, 1244, 1311])
print(decoded_string)
