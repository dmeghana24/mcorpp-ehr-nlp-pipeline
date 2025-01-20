
import medspacy
import pandas as pd

nlp = medspacy.load()
df = pd.read_csv('data/mimic_with_sections.csv')

def has_positive_mention(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PROBLEM" and "embolism" in ent.text:
            if not ent._.negated and not ent._.historical:
                return True
    return False

df['PE_positive'] = df['text'].apply(has_positive_mention)
df.to_csv('data/mimic_context.csv', index=False)
print(df['PE_positive'].value_counts())
