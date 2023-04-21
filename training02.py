import sys

from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

def tokenize_function(examples):
    return tokenizer(examples['text'])

dataset = load_dataset('text', data_files={'train': ['text_file.txt']})

dataset = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(output_dir='./new_model',
                                  overwrite_output_dir=True,
                                  num_train_epochs=3,
                                  per_device_train_batch_size=16,
                                  save_total_limit=2)
# print(len(dataset['train']))
# sys.exit(0)
trainer = Trainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=dataset['train'])

trainer.train()

trainer.save_model('./new_model3')
