This repository contains code for the Trustworthy Trading of Synthetic Data Using Zero-Knowledge Proofs paper.



### Install packages for ZKP and synthetic data generation 
```
pip install -U noknow
pip install ctgan
pip install sdv
pip install synthcity
```

### Datasets

Adult: Census income data (```income>50K``` as target).

Law: Law school data (```first_pf``` as target).

Loan: Bank personal loan data (```Personal Loan``` as target).

Note: Update dataset paths in src/data_loader.py to match your local environment.

## technical motivation folder
This folder contains Python scripts to evaluate synthetic data generation using multiple generators (CTGAN, TVAE, NFLOW) under two types of attacks (Feature Selection Attack and Label Flipping Attack) and a non-attack baseline. The code evaluates synthetic data quality using distribution similarity metrics (Wasserstein Distance, KS Score, KL Divergence) and machine learning model performance (Logistic Regression, Random Forest, MLP).

### Project Structure
```
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
```

## RQ folder
This folder contains Python scripts to verify the integrity and quality of synthetic data generated in an outsourcing scenario, using Schnorr-based Non-Interactive Zero-Knowledge Proofs (NIZKPs). The system evaluates synthetic data under various attack scenarios (data source, quality fraud, privacy leak, distribution tamper) and a non-attack baseline.

### Project Structure
```
synthetic-data-verification/
│   ├── data_processing.py       # Dataset loading, preprocessing, and CTGAN training
│   ├── attacks.py              # Attack simulation logic (data source, quality fraud, privacy leak, distribution tamper)
│   ├── metrics.py              # Quality and privacy metric computation
│   ├── zkp_verification.py     # ZKP client-server verification logic
│   ├── main.py                 # Main script for experiment coordination and result reporting
│   ├── DatasetHashRegistry.txt                 # blockchain
```
