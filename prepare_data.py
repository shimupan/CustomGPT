import torch
import tiktoken

class Data:
    def __init__(self, text):
        self.enc = tiktoken.encoding_for_model("gpt-4")
        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        self.unique_characters = sorted(list(set(text)))
        self.vocabulary_size = len(self.unique_characters)
        self.training_data = None
        self.testing_data = None
        self.__init_data()
    
    def __init_data(self):
        split = int(0.9*len(self.data)) # 90% training data 
        self.training_data = self.data[:split]
        self.testing_data = self.data[split:]

    def encode(self, word):
        return self.enc.encode(word)

    def decode(self, encoding):
        return self.enc.decode(encoding)