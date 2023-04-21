import os.path
import sys

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the trained model from disk
model_path = "/home/richard/disks/sdb6/2023/openalpr/torch/new_model3"
if not os.path.exists(model_path):
    print(model_path, "is not exist")
    sys.exit(0)
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained(model_path)

# Generate an array of text samples
prompt = "Seapapa is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
num_samples = 5
output = model.generate(input_ids, max_length=50, num_return_sequences=num_samples, do_sample=True)

# Decode the generated text and print it to the console
for i in range(num_samples):
    generated_text = tokenizer.decode(output[i], skip_special_tokens=True)
    print(f"Sample {i+1}: {generated_text.strip()}")