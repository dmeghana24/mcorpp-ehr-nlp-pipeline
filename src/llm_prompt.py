
import openai
import pandas as pd
import time

openai.api_key = "sk-..."  # Set your API key

PROMPT = """
Given the following clinical note, does it mention pulmonary embolism (PE), anticoagulation (ANTICOAG), or neither (CONTROL/NEGATIVE)? Return one label: PE, ANTICOAG, NEGATIVE, or CONTROL.

Note: {note}
"""

def llm_label(note):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": PROMPT.format(note=note)}]
    )
    return response['choices'][0]['message']['content'].strip().split()[0]

df = pd.read_csv('data/mimic_labeled.csv')
labels = []
for note in df['text']:
    try:
        labels.append(llm_label(note))
        time.sleep(1)
    except Exception as e:
        labels.append("ERROR")
df['llm_pred'] = labels
df.to_csv('results/llm_preds.csv', index=False)
