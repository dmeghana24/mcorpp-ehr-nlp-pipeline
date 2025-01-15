
import pandas as pd
import re

def label_note(text):
    text = text.lower()
    if re.search(r'no (evidence|signs) of pulmonary embolism', text):
        return 'NEGATIVE'
    elif 'pulmonary embolism' in text:
        return 'PE'
    elif any(x in text for x in ['anticoag', 'heparin', 'warfarin', 'apixaban']):
        return 'ANTICOAG'
    else:
        return 'CONTROL'

df = pd.read_csv('data/mimic_sample.csv')  # swap for mimic_extracted.csv for full set
df['label'] = df['text'].apply(label_note)
df.to_csv('data/mimic_labeled.csv', index=False)
print(df['label'].value_counts())
