# train_gpt2.py

import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Step 1: Load Dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Step 2: Preprocess Data
def preprocess_data(data):
    # We will use the 'description' column of the wine reviews dataset
    data = data[['description']]
    data.columns = ['text']
    return Dataset.from_pandas(data)

# Step 3: Tokenize Data
def tokenize_data(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    
    return dataset.map(tokenize_function, batched=True)

# Step 4: Fine-tune GPT-2 Model
def fine_tune_model(dataset, tokenizer):
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')

# Step 5: Generate Text
def generate_text(prompt, tokenizer, model, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Load and preprocess dataset
    file_path = 'winemag-data_first150k.csv'
    data = load_data(file_path)
    dataset = preprocess_data(data)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Tokenize dataset
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    # Fine-tune the model
    fine_tune_model(tokenized_dataset, tokenizer)

    # Load the fine-tuned model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
    tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')

    # Generate text
    prompt = "The wine's flavor profile is"
    generated_text = generate_text(prompt, tokenizer, model)
    print(generated_text)

