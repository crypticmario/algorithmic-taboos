# algorithmic-taboos

Code and data repository for the Master's Research Project:

**"Algorithmic Taboos: Dialogical Repression and the Differential Treatment
of National Identity in Large Language Models"**

Richard Anekwe | University of Limerick | 2026

Supervisors: Prof. Mike Quayle, Dr. Ana Jovancevic

## Repository Structure

### data_collection/
- Local data collection script (Ollama API, Python 3.10)
- Google Colab notebook for Llama 3.1, Mistral 7B, and Qwen 2.5

### recoding/
- Script to recode responses after correcting the safety lecture
  threshold from 100 to 250 characters and expanding the keyword list

### analysis/
- Statistical analysis: chi-square tests, independent-samples t-tests,
  binary logistic regression (scipy, statsmodels, pandas)

### data/
- Llama 3 8B (600 trials)
- Llama 3.1 (600 trials)
- Mistral 7B (600 trials)
- Qwen 2.5 (600 trials)
- Total: 2,400 trials across 20 national identities and 3 prompt valences

### results/
- Full statistical output from all analyses

## Models Tested
| Model | Developer | Origin |
|-------|-----------|--------|
| Llama 3 8B | Meta | United States |
| Llama 3.1 | Meta | United States |
| Mistral 7B | Mistral AI | France |
| Qwen 2.5 | Alibaba | China |

## Requirements
- Python 3.10
- Ollama (local model hosting)
- pandas, scipy, statsmodels

## Citation
Anekwe, R. (2026). Algorithmic taboos: Dialogical repression and the
differential treatment of national identity in large language models
[Master's thesis, University of Limerick].
