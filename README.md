# Transformer Translation: Arabic to Italian
This repository implements a neural machine translation pipeline for translating text from Arabic to Italian using the `Helsinki-NLP/opus-mt-ar-it` model. The project utilizes the Hugging Face `transformers` library for fine-tuning, training, and deploying the model.

## Project Highlights
- **Dataset**: [Helsinki-NLP/news_commentary](https://huggingface.co/datasets/Helsinki-NLP/news_commentary)
- **Pretrained Model**: [Helsinki-NLP/opus-mt-ar-it](https://huggingface.co/Helsinki-NLP/opus-mt-ar-it)
- **Framework**: Hugging Face Transformers
- **Language Pair**: Arabic (`ar`) to Italian (`it`)
- **Deployment**: Streamlit for user-friendly interaction with the trained model

---

## Features
- Fine-tuning the MarianMT model for Arabic-Italian translation
- Evaluation using BLEU scores
- Real-time translation using a Streamlit web app

---

## Installation
Install the required libraries:
```bash
pip install transformers datasets evaluate sacrebleu numpy streamlit
```

---

## Dataset
The `news_commentary` dataset from Hugging Face is used, specifically the Arabic-Italian (`ar-it`) translation pair. The dataset is split into training and testing subsets (80/20 split).

---

## Model
The `Helsinki-NLP/opus-mt-ar-it` model from Hugging Face is used for fine-tuning. The MarianMT architecture is leveraged for translation tasks.

---

## Training Pipeline

### 1. Load Dataset
```python
from datasets import load_dataset
ds = load_dataset("Helsinki-NLP/news_commentary", "ar-it")
ds = ds.remove_columns('id')
ds = ds['train'].train_test_split(train_size=0.8)
```

### 2. Preprocessing
```python
from transformers import MarianTokenizer
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-it")

def preprocessing(batch):
    inputs = [example['ar'] for example in batch['translation']]
    targets = [example['it'] for example in batch['translation']]
    model_inputs = tokenizer(inputs, text_target=targets, padding=True, return_tensors='pt', truncation=True)
    return model_inputs

ds = ds.map(preprocessing, batched=True)
```

### 3. Training
Fine-tune the model using Hugging Face's `Seq2SeqTrainer`:
```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, MarianMTModel, DataCollatorForSeq2Seq

model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-it")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./Helsinki-mt-ar-it",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    num_train_epochs=3,
    fp16=True,
    warmup_steps=100,
    logging_steps=100,
    save_steps=4000
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
```

### 4. Save and Load Model
Save the fine-tuned model for later use:
```python
trainer.save_model('/kaggle/working/model')
```

---

## Evaluation
BLEU score is computed using the `evaluate` library:
```python
import evaluate
metric = evaluate.load("sacrebleu")
```

---

## Prediction
Use the fine-tuned model to make predictions:
```python
def predict(text, model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    inputs = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=300, do_sample=True, top_k=30, top_p=0.95)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

text = "مرحبا"
print(predict(text, '/kaggle/working/model'))
```

---

## Streamlit App
A Streamlit app is provided for real-time interaction with the model:
```python
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

model_path = '/kaggle/working/model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
translation_pipeline = pipeline('translation', model=model, tokenizer=tokenizer)

st.title("Arabic to Italian Translator")

input_text = st.text_area("Enter text to translate:")
if st.button("Translate"):
    if input_text:
        translated_text = translation_pipeline(input_text)
        st.write("Translation:", translated_text[0]['translation_text'])
    else:
        st.write("Please enter some text.")
```

Run the app:
```bash
streamlit run app.py
```

---

## Repository Structure
- `train.py`: Script for fine-tuning the model
- `app.py`: Streamlit app for deploying the model
- `model/`: Directory containing the fine-tuned model

---

## Results
The model achieves a BLEU score of `XX.XX` on the test dataset (adjust based on actual results). It demonstrates effective translation from Arabic to Italian.

---

## Contributing
Feel free to fork this repository and submit pull requests. Contributions are welcome!

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---
