This repository contains code for the Trustworthy Exchange of Synthetic Data Through Zero-Knowledge Proof paper.



### Install packages for ZKP and synthetic data generation 
```
pip install -U noknow
pip install ctgan
pip install sdv
pip install synthcity
```

## RQ1 folder
This folder contains Python scripts to evaluate synthetic data generation using multiple generators (CTGAN, TVAE, NFLOW) under two types of attacks (Feature Selection Attack and Label Flipping Attack) and a non-attack baseline. The code evaluates synthetic data quality using distribution similarity metrics (Wasserstein Distance, KS Score, KL Divergence) and machine learning model performance (Logistic Regression, Random Forest, MLP).

### Project Structure
synthetic-data-evaluation/
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading and dataset definitions
│   ├── preprocessing.py         # Data preprocessing (encoding, one-hot, etc.)
│   ├── metrics.py               # Distribution similarity metrics
│   ├── generators.py            # Synthetic data generators (CTGAN, TVAE, NFLOW)
│   ├── models.py                # ML models
│   ├── feature_selection_attack.py  # Feature selection attack logic
│   ├── label_flipping_attack.py     # Label flipping attack logic
│   ├── main_feature_selection.py    # Main script for feature selection attack
│   ├── main_label_flipping.py      # Main script for label flipping attack
│   ├── main_synthetic_evaluation.py  # Main script for non-attack synthetic evaluation
