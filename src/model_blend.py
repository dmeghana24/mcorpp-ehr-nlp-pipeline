import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report

bert = pd.read_csv('results/clinicalbert_preds.csv')
llm = pd.read_csv('results/llm_preds.csv')
true = bert['label']
bert_pred = bert['pred_label']
llm_pred = llm['llm_pred']

# Simple ensemble: majority vote with tiebreaker logic
ensemble = []
for b, l in zip(bert_pred, llm_pred):
    if b == l: ensemble.append(b)
    elif b == "PE": ensemble.append(b)
    elif l == "ANTICOAG": ensemble.append(l)
    else: ensemble.append("CONTROL")

print("ClinicalBERT:\n", classification_report(true, bert_pred))
print("LLM:\n", classification_report(true, llm_pred))
print("Ensemble:\n", classification_report(true, ensemble))

# GroupKFold CV by subject_id (no leakage)
df = pd.read_csv('data/mimic_labeled.csv')
groups = df['subject_id']
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(gkf.split(df, df['label'], groups)):
    print(f"Fold {fold}: train {len(train_idx)}, test {len(test_idx)}")
    # Add model training/inference loop per fold if desired

pd.DataFrame({'true': true, 'bert': bert_pred, 'llm': llm_pred, 'ensemble': ensemble}).to_csv('results/model_comparison.csv', index=False)

