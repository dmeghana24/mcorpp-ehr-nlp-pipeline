# mcorpp-ehr-nlp-pipeline
# MCORRP-Inspired EHR NLP Pipeline

A modular, pipeline for extracting structured clinical data (e.g., pulmonary embolism, anticoagulation) from unstructured EHR free-text using BERT, LLMs, section segmentation, context detection, and advanced evaluation. Real MIMIC-III pipeline ready.

- **No patient data included.** All scripts run on MIMIC-III for credentialed users only.
- For educational/demo purposes. 


# MCORRP-Inspired EHR NLP Pipeline

## Features
- **MIMIC-III extraction** with weak labeling
- **Section and context detection** (negation, temporality)
- **Note-level & mention-level classification** (BERT, LLM, NER)
- **Cross-validation by patient**
- **Ensembling & evaluation** (AUROC, PR, confusion matrix, calibration)
- **Explainability** (SHAP/attention viz)
- **Streamlit dashboard**

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Extract your own MIMIC notes: `python src/extract_mimic_notes.py`
3. Label notes: `python src/label_mimic_notes.py`
4. Section splitting: `python src/section_splitter.py`
5. Context/negation: `python src/context_negation.py`
6. NER: `python src/ner_pipeline.py`
7. Train/eval models: `python src/clinicalbert_baseline.py`, `python src/llm_prompt.py`
8. Model blending/evaluation: `python src/model_blend.py`, `python src/validation.py`
9. Explainability: `python src/explainability.py`
10. Dashboard: `streamlit run src/dashboard.py`


The repo includes a 5-row synthetic dataset for demo/testing. For real EHR data, see MIMIC-III setup instructions.

## For MIMIC use: You must have access to the database. No data is distributed here.

## 🩺 Use Cases

- **Pulmonary Embolism Cohort Identification:** Detecting patients with PE and related risk factors from radiology and clinical notes.
- **Anticoagulation Quality Improvement:** Extracting indications, dosages, and adherence from medication and progress notes.
- **Antimicrobial Stewardship:** Flagging mentions of inappropriate prescribing and tracking intervention outcomes.
- **General Clinical Research:** Rapid EHR abstraction for building high-quality datasets to support hypothesis generation and outcomes analysis.

---

## 📂 Sample Data

Sample data (de-identified and synthetic) are provided in the `/sample_data/` directory:
- `sample_notes.csv` – Example EHR note excerpts for processing.
- `sample_output.csv` – Example annotated outputs (entities, labels, categories).

**Note:** All data provided are simulated and free of PHI.

---

## Workflow

```mermaid
flowchart TD
    A[Input EHR Notes] --> B[Text Preprocessing]
    B --> C[Sentence Segmentation & Tokenization]
    C --> D[Entity Recognition (Regex + CRF)]
    D --> E[Negation & Context Detection]
    E --> F[Label Assignment & Categorization]
    F --> G[Structured Output (CSV/JSON)]
    G --> H[QC, Error Analysis, Cohort Filtering]

```

## How to Run

**Clone this repo:**
```bash
git clone https://github.com/dmeghana24/mcorpp-ehr-nlp-pipeline.git
cd mcorpp-ehr-nlp-pipeline
```
## 💻 Install Dependencies

See `requirements.txt` or set up the environment as described in `/docs/`.

---

## 📝 Process Sample Notes

- Run `main.py` or use the provided Jupyter notebooks.
- Output files will be generated in `/output/`.

---

## 📊 Example

**Input:**  
"Patient with history of DVT, started on apixaban. No evidence of PE on CT scan. Antibiotics discontinued."

**Output (sample):**

| Entity       | Category         | Negation | Context         |
|--------------|------------------|----------|-----------------|
| DVT          | Condition        | False    | History         |
| apixaban     | Medication       | False    | Current         |
| PE           | Condition        | True     | Imaging Result  |
| Antibiotics  | Medication Class | True     | Current         |

---

## 📑 Documentation

- Full documentation of pipeline modules and configuration: [`docs/`](./docs/)
- Example patterns, rules, and CRF features: [`patterns/`](./patterns/)
- Error analysis and cohort review examples: [`output/error_analysis/`](./output/error_analysis/)

---

## 📢 Collaboration & Support

Some code and data are abridged or synthetic due to data-sharing and IRB constraints.  
For full code, real data integrations, or research collaboration, please contact:  
📧 **dmeghana@umich.edu**

---

## ✨ Citation

If you use or adapt this pipeline, please cite:

> D. Meghana, 2024. NLP pipeline for EHR abstraction and cohort identification at MCORRP. [GitHub Repository](https://github.com/dmeghana24/mcorpp-ehr-nlp-pipeline)


## References

- Johnson, A. E. W., et al. (2016). [MIMIC-III, a freely accessible critical care database](https://www.nature.com/articles/sdata201635). Scientific Data.
- Alsentzer, E., et al. (2019). [Publicly Available Clinical BERT Embeddings](https://arxiv.org/abs/1904.03323).
- Vaswani, A., et al. (2017). [Attention is All You Need](https://arxiv.org/abs/1706.03762). (transformers)
- medspaCy: [https://github.com/medspacy/medspacy](https://github.com/medspacy/medspacy)
- SHAP: Lundberg, S. M., & Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874).
- LLM API: [OpenAI GPT-3.5/4](https://openai.com/research)
