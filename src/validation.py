
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize

df = pd.read_csv('results/model_comparison.csv')
y_true = df['true']
y_bert = df['bert']
labels = sorted(set(y_true))
y_true_bin = label_binarize(y_true, classes=labels)
y_bert_bin = label_binarize(y_bert, classes=labels)

roc_auc = roc_auc_score(y_true_bin, y_bert_bin, average="micro")
print("Micro-average AUROC:", roc_auc)

for i, label in enumerate(labels):
    precision, recall, _ = precision_recall_curve(y_true_bin[:,i], y_bert_bin[:,i])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"{label} AUPR={pr_auc:.2f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
plt.title("Precision-Recall Curves (ClinicalBERT)")
plt.savefig("figures/pr_curve.png")
plt.close()

cm = confusion_matrix(y_true, y_bert, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.title("ClinicalBERT Confusion Matrix")
plt.savefig("figures/confusion_matrix.png")
plt.close()
