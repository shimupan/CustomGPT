import tiktoken
from prepare_data import Data

text = None
with open("test.txt","r",encoding="utf-8") as file:
    text = file.read()

data = Data(text)

print(data.unique_characters)
print(data.vocabulary_size)
print(data.data[:100])