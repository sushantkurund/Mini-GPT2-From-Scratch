from src.tokenizer import CharacterTokenizer

text = "hello world"

tokenizer = CharacterTokenizer(text)

encoded = tokenizer.encode("hello")
decoded = tokenizer.decode(encoded)

print("Vocabulary Size:", tokenizer.vocab_size)
print("Encoded:", encoded)
print("Decoded:", decoded)