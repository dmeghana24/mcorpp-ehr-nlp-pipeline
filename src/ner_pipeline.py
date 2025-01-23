
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

def get_entities(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).logits
    preds = outputs.argmax(-1).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    entities = []
    for token, pred in zip(tokens, preds):
        if pred != 0:  # 0 is usually 'O'
            entities.append((token, model.config.id2label[pred]))
    return entities

df = pd.read_csv('data/mimic_context.csv')
df['entities'] = df['text'].apply(get_entities)
df.to_csv('data/mimic_with_ner.csv', index=False)
