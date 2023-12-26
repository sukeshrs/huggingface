from transformers import pipeline
from timeit import default_timer as timer

start = timer()
classifier = pipeline("sentiment-analysis")
output = classifier("she is playing with food")
end = timer()
print(end - start)
print(output)

