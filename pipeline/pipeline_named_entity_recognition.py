from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
output = ner("My name is Peter. I am doctor.I am travelling to New York for Work.")

print(output)
