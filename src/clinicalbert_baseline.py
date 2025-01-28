
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/mimic_labeled.csv')
label2id = {k: v for v, k in enumerate(df['label'].unique())}
id2label = {v: k for k, v in label2id.items()}
df['target'] = df['label'].map(label2id)

train, test = train_test_split(df, stratify=df['label'], test_size=0.2, random_state=42)
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
def tokenize(batch): return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

train_encodings = tokenize(train)
test_encodings = tokenize(test)

class NotesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_ds = NotesDataset(train_encodings, train['target'])
test_ds = NotesDataset(test_encodings, test['target'])

model = AutoModelForSequenceClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT", num_labels=len(label2id)
)
training_args = TrainingArguments(
    output_dir='./results', num_train_epochs=2, per_device_train_batch_size=8, evaluation_strategy="epoch"
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds)
trainer.train()

preds = trainer.predict(test_ds)
test['pred'] = preds.predictions.argmax(axis=1)
test['pred_label'] = test['pred'].map(id2label)
test.to_csv('results/clinicalbert_preds.csv', index=False)





