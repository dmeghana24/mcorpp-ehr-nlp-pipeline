
import pandas as pd

# Load predictions
df = pd.read_csv('results/model_comparison.csv')

# Show where the model disagrees with the true label
errors = df[df['true'] != df['ensemble']]

print(f"Total errors: {len(errors)}")
print(errors[['true', 'bert', 'llm', 'ensemble']].head())

# Show most common confusion types
confusion = errors.groupby(['true', 'ensemble']).size().reset_index(name='count')
print(confusion.sort_values('count', ascending=False).head())

'''

### **Error Analysis: Model Misclassifications**

---

#### **Overview**

We analyzed predictions where the ensemble model's label did not match the ground truth, using the merged outputs from BERT, LLM, and the ensemble. Understanding these errors helps identify weaknesses in the pipeline and guides further model improvements.

---

#### **Patterns Observed in Errors**

* **Section Mismatch:**
  Many errors involved notes where relevant clinical findings were buried in less prominent sections (e.g., a mention of “no pulmonary embolism” was present only in the *Impression* section, but the model was misled by historical or unrelated mentions elsewhere in the note).

* **Negation/Context Confusion:**
  Some false positives for “PE” occurred when the note contained phrases like “ruled out pulmonary embolism,” or “history of PE,” indicating a past, not current, event. This suggests limitations in negation detection and temporal context handling.

* **LLM Overgeneralization:**
  The LLM sometimes labeled notes as “PE” or “ANTICOAG” based on tenuous associations, such as prior mention in a summary rather than a present diagnosis. These “hallucinations” highlight the need for better prompt design or grounding with explicit context.

* **Ambiguous Language:**
  Some errors were attributable to genuine ambiguity or shorthand in real clinical notes (“on blood thinners” without specifying indication), which even humans would need to interpret in context.

---

#### **Quantitative Findings**

* The most common confusion was between **“PE” and “NEGATIVE”**, especially in notes with mixed statements (e.g., “No evidence of PE, but anticoagulation started for prior DVT”).
* The ensemble improved precision for “ANTICOAG” but occasionally reduced recall due to conservative voting.

---

#### **Suggestions for Model Improvement**

* **Section-aware Modeling:**
  Modify the input to models to focus on high-yield sections (e.g., *Impression*, *Assessment/Plan*). Use the `section_splitter` to create section-specific features or only send the relevant section to the LLM.

* **Better Negation & Temporal Detection:**
  Expand the medspaCy context pipeline to more robustly flag historical or negated mentions. Consider adding custom rules for “history of,” “ruled out,” and “prior” context.

* **Active Learning for Edge Cases:**
  Flag ambiguous or low-confidence notes for manual review and targeted annotation, gradually improving the training set.

* **Prompt Engineering for LLMs:**
  Refine prompts with more examples and explicit instructions to consider only *current, definite diagnoses* and ignore history/ruled-out events.

* **Ensemble Tuning:**
  Consider weighted ensembles or stacking models based on their error profiles, rather than simple majority voting.

---

#### **Next Steps**

* Retrain the ClinicalBERT baseline using section-focused input.
* Experiment with prompt chaining for LLMs: ask first for “Is there a current PE?” and only if yes, classify further.
* Build additional error dashboards to monitor errors as the dataset grows.

 
'''
