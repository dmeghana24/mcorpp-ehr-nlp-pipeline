# mcorpp-ehr-nlp-pipeline
# MCORRP-Inspired EHR NLP Pipeline

A modular, graduate-level pipeline for extracting structured clinical data (e.g., pulmonary embolism, anticoagulation) from unstructured EHR free-text using BERT, LLMs, section segmentation, context detection, and advanced evaluation. Real MIMIC-III pipeline ready.

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


## References

- Johnson, A. E. W., et al. (2016). [MIMIC-III, a freely accessible critical care database](https://www.nature.com/articles/sdata201635). Scientific Data.
- Alsentzer, E., et al. (2019). [Publicly Available Clinical BERT Embeddings](https://arxiv.org/abs/1904.03323).
- Vaswani, A., et al. (2017). [Attention is All You Need](https://arxiv.org/abs/1706.03762). (transformers)
- medspaCy: [https://github.com/medspacy/medspacy](https://github.com/medspacy/medspacy)
- SHAP: Lundberg, S. M., & Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874).
- LLM API: [OpenAI GPT-3.5/4](https://openai.com/research)
