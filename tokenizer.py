from transformers import GPT2Tokenizer

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Define an input text
input_text = 'This is a sample sentence.'

# Tokenize the input text
encoded_input = tokenizer.encode(input_text)

# Print the encoded input
print(encoded_input)