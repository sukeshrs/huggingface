from transformers import pipeline

generator = pipeline("text-generation")
output = generator("In this course, we will teach you how to")

print(output)