# Week 9–10 Progress Report

## Project definition
Predict whether an online shopping session ends in a purchase using the binary target `Revenue` (False = no purchase, True = purchase).

- **Task:** binary classification
- **Primary metric:** macro-F1, because it weights both classes equally under class imbalance.
- **Secondary metrics:** per-class precision/recall/F1, purchase recall, ROC-AUC, accuracy, and confusion matrix.
- **Dataset:** [UCI Online Shoppers Purchasing Intention, ID 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention).

## Week 9 — Data engineering
The pipeline reports missing values and exact duplicates, removes exact duplicate rows, and removes rows with missing target labels. Missing numeric feature values are imputed with the training-set median; missing categorical feature values are imputed with the training-set most frequent value. StandardScaler and OneHotEncoder are also fitted on the training partition only.

The data is split into stratified train/validation/test partitions (70/15/15, random seed 42). This is session-level tabular data and does not provide a person/source grouping key or a chronological sequence key for a forecasting task, so stratified random splitting is used. The held-out test set is not used for training or model selection.

Class imbalance is addressed with balanced class weights computed from training labels only. The validation ablation compares logistic regression without class weights and with `class_weight="balanced"`, reporting per-class recall.

## Week 10 — EDA, baselines, and Model A
The EDA notebook reviews dimensions, data types, missingness, duplicates, target distribution, numerical distributions, and numeric correlations.

The required baselines are evaluated on validation data:
1. **Trivial baseline:** `DummyClassifier(strategy="most_frequent")`.
2. **Learned baseline:** logistic regression.

The unweighted and class-weighted logistic regression comparison is the class-imbalance ablation. Model A is a feed-forward neural network with two hidden layers (64 and 32 units), dropout, early stopping, and class weights derived from the training partition. It is trained after the baselines and compared using validation macro-F1 and per-class recall.

## Reproduction
From the repository root:

```bash
python -m pip install -r requirements.txt
python data/download_data.py
python data/inspect_data.py
python data/clean_data.py
python -m src.run_week10
```

The runner performs preprocessing, baseline training/evaluation, and Model A training/validation. It does not use the test split.

After all model choices are fixed, run the held-out test evaluation once:

```bash
python -m src.evaluate
```

This selects the candidate with the highest validation macro-F1 and then evaluates it on the test split once.

## Validation Results

All models were evaluated on the validation set. The test set was not used for training, model selection, or validation comparisons.

### Validation Performance Comparison

| Model | Accuracy | Macro-F1 | No-Purchase Recall | Purchase Recall |
|---|---:|---:|---:|---:|
| Majority Class Dummy | 0.8438 | 0.4576 | 1.0000 | 0.0000 |
| Unweighted Logistic Regression | 0.8886 | 0.7279 | 0.9819 | 0.3846 |
| Class-Weighted Logistic Regression | 0.8433 | 0.7552 | 0.8550 | 0.7797 |
| Model A Neural Network | 0.8465 | 0.7687 | 0.8453 | 0.8531 |

### Baseline Comparison

The majority-class Dummy Classifier achieved an accuracy of 84.38%, but its Purchase Recall was 0.00%. It predicted every session as No Purchase and failed to identify any actual purchases. This demonstrates why accuracy alone is not sufficient for this imbalanced classification problem.

Unweighted Logistic Regression improved substantially over the Dummy Classifier, achieving a Macro-F1 score of 0.7279 and a Purchase Recall of 38.46%.

### Class Imbalance Ablation

Applying `class_weight="balanced"` to Logistic Regression increased Purchase Recall from 38.46% to 77.97%, an improvement of 39.51 percentage points.

However, No-Purchase Recall decreased from 98.19% to 85.50%, and Purchase Precision decreased from 79.71% to 49.89%. This shows the trade-off introduced by class weighting: the model identifies more purchasing sessions but also produces more false positive purchase predictions.

The Macro-F1 score increased from 0.7279 to 0.7552, indicating a better balance between the two classes.

### Model A Comparison

Model A, a feed-forward neural network, achieved the highest validation Macro-F1 score of 0.7687, outperforming both logistic regression candidates.

It also achieved a Purchase Recall of 85.31% and a No-Purchase Recall of 84.53%. Compared with class-weighted Logistic Regression, Model A improved Macro-F1 from 0.7552 to 0.7687 and Purchase Recall from 77.97% to 85.31%.

Model A was trained for 28 epochs, with early stopping monitoring validation loss and restoring the best weights. The measured training time was 6.18 seconds.

### Validation Model Selection

Model A was selected as the final candidate because it achieved the highest validation Macro-F1 among all evaluated models. The held-out test set remained untouched until the one-time final evaluation reported in the next section.

Copy the measured macro-F1 and both class recalls for each validation model into the final submitted report. Explain whether logistic regression improves over the majority baseline, whether Model A improves over the learned baseline, and the purchase/no-purchase recall trade-off from class weighting. Do not invent results or tune based on test metrics.

## Final Test Evaluation

After completing model selection using the validation set, the final model was selected based on the highest validation Macro-F1 score. The Feed-Forward Neural Network (Model A) achieved the highest validation Macro-F1 of **0.7687**, outperforming both logistic regression baselines.

The selected model was evaluated once on the held-out test set, which was not used during training or model selection.

### Final Test Results

| Metric | Test Result |
|---|---:|
| Accuracy | 0.8504 |
| Macro-F1 | 0.7716 |
| Weighted F1 | 0.8638 |
| ROC-AUC | 0.9194 |
| Purchase Precision | 0.5128 |
| Purchase Recall | 0.8427 |
| Purchase F1-score | 0.6376 |

### Confusion Matrix

Rows represent actual classes and columns represent predicted classes. Class order: No Purchase (0), Purchase (1).

| Actual / Predicted | No Purchase | Purchase |
|---|---:|---:|
| No Purchase | 1316 | 229 |
| Purchase | 45 | 241 |

### Results Interpretation

The neural network achieved a test Macro-F1 score of **0.7716** and an accuracy of **85.04%**. Its ROC-AUC score of **0.9194** indicates strong ability to distinguish between purchasing and non-purchasing sessions.

For the minority Purchase class, the model achieved a recall of **84.27%**, correctly identifying 241 out of 286 actual purchasing sessions. However, its purchase precision was **51.28%**, meaning that some sessions predicted as purchases did not result in actual purchases. This reflects a trade-off between detecting more potential purchases and producing false positive predictions.

The validation and test Macro-F1 scores were close (0.7687 and 0.7716, respectively), with no obvious performance drop on the held-out test set.

### Final Model Selection

Model A was selected because it achieved the highest validation Macro-F1 among the evaluated candidates:

- Unweighted Logistic Regression: 0.7279
- Class-weighted Logistic Regression: 0.7552
- Model A Neural Network: 0.7687

The test set was reserved for the final evaluation and was not used for further model tuning.
