# Anti-Fraud SMS Detection (NLP + OSINT Features)

#### Project Goal

Build a simple but structured ML pipeline for detecting fraudulent SMS messages.

The system combines:
- NLP-based features (TF-IDF)
- OSINT-inspired rule features (URLs, phones, suspicious words)
- Logistic Regression baseline
- Hybrid ML + rule-based risk scoring
- Batch scoring script

This project simulates a simplified fraud detection system similar to those used in fintech or telecom companies.

## Dataset

**SMS Spam Collection Dataset**

- 5,574 SMS messages
- Labels:

  - `ham` (legitimate message)
  - `spam` (fraudulent message)

The dataset contains raw SMS text and binary labels.

## Exploratory Data Analysis

- Checked class imbalance
- Analyzed message length
- Measured frequency of URLs and phone numbers
- Identified words frequently appearing in spam

##### Class Imbalance

Spam messages represent ~13–15% of the dataset.

To handle imbalance:

- Used `stratified train/test split`
- Applied `class_weight='balanced'` in Logistic Regression

---

## Feature Engineering

#### 1. Text Preprocessing

- Lowercasing
- URL masking
- Phone masking
- Number normalization
- Punctuation removal


#### 2. NLP Features

Used **TF-IDF vectorization**:

- 1–2 grams
- max 2000 features
- English stop-words removal
- min document frequency filtering


#### 3. OSINT-Inspired Features

| Feature             | Description              |
| ****************--- | ************************ |
| url_count           | Number of URLs           |
| phone_count         | Number of phone patterns |
| digit_count         | Number of digits         |
| caps_ratio          | Uppercase letter ratio   |
| money symbols       | $, €, £                  |
| fraud keyword count | Spam-specific words      |

These features simulate rule-based risk indicators used in anti-fraud systems.


## Model

#### Baseline Model

**Logistic Regression**

Why:
- Interpretable
- Fast
- Good baseline for text classification
- Common in production systems

## Evaluation Metrics

Used the following metrics:

- ROC-AUC
- Precision
- Recall
- F1-score
- PR-AUC
- Confusion Matrix
- 5-fold Cross-Validation

Fraud detection focus:
- High recall (to reduce missed fraud)
- Balanced precision


## Hybrid Risk Scoring

In addition to ML model, implemented a simple rule-based scoring system.

##### Rule Examples

- Contains URL → +2
- Contains phone → +2
- Many digits → +1
- Many fraud keywords → +2
- Contains money symbols → +2

Final hybrid score:

```
final_score = 0.7 - ML_probability + 0.3 - normalized_rule_score
```

Threshold selected using F1 optimization on PR curve.
This approach improves interpretability and simulates real-world fraud risk scoring.


##### Pipeline Steps

1. Load data
2. Preprocess text
3. Extract features
4. Train model
5. Evaluate
6. Save pipeline
7. Run batch scoring


## Production-Oriented Design

- Used `sklearn Pipeline`
- Avoided data leakage
- Stratified split
- Saved full pipeline as single artifact
- Implemented batch scoring script

Output contains:
- probability
- fraud_flag


## Key Insights

- URLs strongly correlate with spam
- Fraud messages contain more digits
- Certain words (e.g. win, urgent, verify) are strong signals
- Hybrid ML + rules improves transparency


## Tech Stack

- Python
- pandas
- scikit-learn
- TF-IDF
- regex
- joblib