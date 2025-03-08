# GBDT_Catalyst

##Catalyst Property Prediction with GBDT

This repository implements Gradient Boosting Decision Trees (GBDT) for modeling relationships between catalyst features and target properties (e.g. activity/stability). The model captures non-linear interactions through iterative decision tree ensembles, providing interpretable feature importance analysis.

### Key Function
Feature Engineering: Automated handling of catalyst descriptors (electronic/geometric features).
Non-linear Modeling: Capture complex feature-property relationships via multi-level tree splits
Feature importance: Feature importance value integration for feature contribution (see `GBDT_catalyst.py`)

### Installation
Install the latest stable version: pip install gbdt

### Code Structure
```
├── GBDT_catalyst.py     # Main model class and training pipeline
├── data                 # Dataset directory
│   ├── Train_Data.csv   # Training set (features & labels)
│   └── Test_Data.csv    # Test set (features & labels)
```
