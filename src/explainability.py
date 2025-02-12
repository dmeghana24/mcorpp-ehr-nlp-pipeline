
import shap
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

df = pd.read_csv('data/mimic_labeled.csv').sample(10)

def get_features(texts):
    return tokenizer(texts, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

background = df['text'][:5].tolist()
explainer = shap.Explainer(
    lambda x: model(**get_features(x)).logits.softmax(dim=1).detach().numpy(),
    background
)

shap_values = explainer(df['text'][5:].tolist())
shap.plots.text(shap_values)
plt.savefig("figures/attention_example.png")
