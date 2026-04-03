# Introduction to Machine Learning with Python

> **Book:** *Introduction to Machine Learning with Python — A Guide for Data Scientists*  
> **Authors:** Andreas C. Müller & Sarah Guido  

---

## 📚 Repository Overview

This repository contains **Jupyter Notebook reproductions** of all 8 chapters of *Introduction to Machine Learning with Python*. 

---

## 📁 Repository Structure

```
Introduction-to-Machine-Learning-with-Python/
│
├── README.md
├── Chapter_01_Introduction.ipynb
├── Chapter_02_Supervised_Learning.ipynb
├── Chapter_03_Unsupervised_Learning_and_Preprocessing.ipynb
├── Chapter_04_Representing_Data_and_Feature_Engineering.ipynb
├── Chapter_05_Model_Evaluation_and_Improvement.ipynb
├── Chapter_06_Algorithm_Chains_and_Pipelines.ipynb
├── Chapter_07_Working_with_Text_Data.ipynb
└── Chapter_08_Wrapping_Up.ipynb
```

---

## 🗂️ Chapter Summaries

---

### Chapter 1 — Introduction
**File:** `Chapter_01_Introduction.ipynb` | **Book pages:** 1–24

Lays the conceptual and practical foundation for machine learning with Python.

**Topics:** Why ML? • Types of ML • Core Python libraries • Iris dataset • k-Nearest Neighbors • Train/Test Split

| Key Concept | Takeaway |
|---|---|
| Machine Learning | Learns patterns from data — no handcoded rules |
| Supervised Learning | Learns from labeled input/output pairs |
| Training Set | Data used to fit the model |
| Test Set | Data used to evaluate generalization — never seen during training |
| k-NN | Predicts by majority vote of k nearest training examples |
| `sklearn` API | `fit()` → `predict()` → `score()` — same for ALL models |

**Result:** ~97% accuracy on the Iris test set in 5 lines of code.

---

### Chapter 2 — Supervised Learning
**File:** `Chapter_02_Supervised_Learning.ipynb` | **Book pages:** 25–128

The largest chapter — covers 8 major supervised learning algorithms with full theory, implementation, and comparison.

**Topics:** Overfitting & Underfitting • Classification vs Regression • 8 Algorithms

| Algorithm | Task | Key Strength | Needs Scaling? |
|---|---|---|---|
| **k-Nearest Neighbors** | Cls + Reg | Simple baseline | ✓ |
| **Linear / Logistic Regression** | Cls + Reg | Interpretable, fast | ✓ |
| **Ridge / Lasso** | Reg | Regularized, feature selection (Lasso) | ✓ |
| **Naive Bayes** | Cls | Extremely fast, great for text | ✗ |
| **Decision Trees** | Cls + Reg | Fully interpretable | ✗ |
| **Random Forest** | Cls + Reg | Robust default, excellent baseline | ✗ |
| **Gradient Boosting** | Cls + Reg | Often best accuracy | ✗ |
| **Support Vector Machines** | Cls + Reg | Powerful in high dimensions | ✓ |
| **Neural Networks (MLP)** | Cls + Reg | Learns complex patterns | ✓ |

**Key Insights:**
- Always scale features for k-NN, SVM, Logistic Regression, and Neural Networks
- Lasso performs automatic feature selection (sparse coefficients)
- Gradient Boosting often wins but requires careful hyperparameter tuning
- Use `predict_proba()` to get confidence scores, not just hard predictions

---

### Chapter 3 — Unsupervised Learning and Preprocessing
**File:** `Chapter_03_Unsupervised_Learning_and_Preprocessing.ipynb` | **Book pages:** 129–212

Covers ML without labels — preprocessing, dimensionality reduction, and clustering.

#### Preprocessing & Scaling

| Technique | Formula | Best For |
|---|---|---|
| **StandardScaler** | (x − μ) / σ | Roughly Gaussian data |
| **MinMaxScaler** | (x − min) / (max − min) | Fixed [0,1] range needed |
| **RobustScaler** | (x − Q₂) / (Q₃ − Q₁) | Data with outliers |

> **Rule:** Always `fit` scalers on training data only — fitting on test data causes **data leakage**!

#### Dimensionality Reduction

| Technique | Type | Use Case |
|---|---|---|
| **PCA** | Linear | Preprocessing, noise reduction, any data |
| **NMF** | Linear (non-negative) | Text topics, image parts |
| **t-SNE** | Non-linear | **Visualization ONLY** |

#### Clustering

| Algorithm | Strengths | Weaknesses |
|---|---|---|
| **k-Means** | Fast, scalable | Needs k, assumes spherical clusters |
| **Agglomerative** | Hierarchical view | Cannot predict new points |
| **DBSCAN** | Arbitrary shapes, handles noise | Needs eps/min_samples tuning |

---

### Chapter 4 — Representing Data and Feature Engineering
**File:** `Chapter_04_Representing_Data_and_Feature_Engineering.ipynb` | **Book pages:** 213–258

Feature engineering — how you represent data — can matter more than algorithm choice.

**Topics:** Categorical Encoding • Binning • Polynomial Features • Log Transforms • Feature Selection

| Technique | Tool | When to Use |
|---|---|---|
| **One-Hot Encoding** | `OneHotEncoder`, `pd.get_dummies` | Nominal categories (no order) |
| **Ordinal Encoding** | `OrdinalEncoder` | Ordered categories |
| **ColumnTransformer** | `ColumnTransformer` | Different transforms per column type |
| **Binning** | `KBinsDiscretizer` | Helps linear models with non-linear features |
| **Polynomial Features** | `PolynomialFeatures` | Allow linear models to fit curves |
| **Log Transform** | `np.log` | Right-skewed data (counts, prices, income) |
| **Univariate Selection** | `SelectKBest` | Keep top k most informative features |
| **Model-based Selection** | `SelectFromModel` | Use Random Forest importance scores |

**Key Insight:** Tree-based models handle raw features well; linear models benefit greatly from encoding and polynomial features.

---

### Chapter 5 — Model Evaluation and Improvement
**File:** `Chapter_05_Model_Evaluation_and_Improvement.ipynb` | **Book pages:** 259–318

Building a model is only half the work — evaluating it correctly is equally critical.

**Topics:** Cross-Validation • Grid Search • Confusion Matrix • Precision/Recall/F1 • ROC-AUC

#### Cross-Validation Strategies

| Strategy | When to Use |
|---|---|
| **k-Fold (k=5 or 10)** | Default choice |
| **Stratified k-Fold** | Classification with imbalanced classes |
| **ShuffleSplit** | Large datasets, fast random splits |

#### Classification Metrics

| Metric | Formula | Best When |
|---|---|---|
| **Accuracy** | (TP+TN) / total | Balanced classes |
| **Precision** | TP / (TP+FP) | Minimizing false positives (spam) |
| **Recall** | TP / (TP+FN) | Minimizing false negatives (disease) |
| **F1-Score** | 2·P·R / (P+R) | Both precision and recall matter |
| **ROC-AUC** | Area under ROC | Overall ranking quality |

**Critical Rule:** The test set must be used **exactly once** — at the very end. All tuning uses cross-validation on training data only.

---

### Chapter 6 — Algorithm Chains and Pipelines
**File:** `Chapter_06_Algorithm_Chains_and_Pipelines.ipynb` | **Book pages:** 319–336

Real ML workflows chain multiple steps. Doing this manually creates data leakage. `Pipeline` solves this elegantly.

**Topics:** Data Leakage • `Pipeline` • `make_pipeline` • Pipelines + GridSearch • `ColumnTransformer`

```python
# The Production-Ready ML Template:
pipe = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(),  numeric_cols),
        ('cat', OneHotEncoder(),   cat_cols),
    ])),
    ('model', GradientBoostingClassifier())
])
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
final_score = grid.score(X_test, y_test)  # Use test set ONCE
```

**Parameter naming:** `stepname__parametername` (e.g., `svm__C`, `model__n_estimators`)

---

### Chapter 7 — Working with Text Data
**File:** `Chapter_07_Working_with_Text_Data.ipynb` | **Book pages:** 337–370

Text is one of the most abundant data forms — but ML needs numeric inputs.

**Topics:** Bag of Words • CountVectorizer • TF-IDF • Stop Words • n-Grams • Topic Modeling (LDA)

| Technique | Tool | Key Idea |
|---|---|---|
| **Bag of Words** | `CountVectorizer` | Count word occurrences per document |
| **TF-IDF** | `TfidfVectorizer` | Downweight common words, upweight rare ones |
| **Stop words** | `stop_words='english'` | Remove uninformative words |
| **min_df** | `min_df=5` | Ignore very rare words (reduce noise) |
| **n-Grams** | `ngram_range=(1,2)` | Capture word pairs ("not good" ≠ "good") |
| **LDA** | `LatentDirichletAllocation` | Discover hidden topics in a corpus |

**Best text classification stack:**
```python
Pipeline([
    ('tfidf', TfidfVectorizer(min_df=3, ngram_range=(1,2), stop_words='english')),
    ('clf',   LogisticRegression(C=1, max_iter=2000))
])
```

---

### Chapter 8 — Wrapping Up
**File:** `Chapter_08_Wrapping_Up.ipynb` | **Book pages:** 371–392

Synthesizes everything into a practical framework for real-world ML problems.

**Topics:** ML Problem Framing • Algorithm Selection Guide • Full Benchmark • Resources

#### The 10 Golden Rules of Machine Learning:
1. **Start simple** — try a linear model before anything else
2. **Always split first** — train/test before any processing
3. **Use Pipelines** — prevent data leakage automatically
4. **Choose the right metric** — accuracy misleads on imbalanced data
5. **Cross-validate everything** — one split is not enough
6. **Scale features** for k-NN, SVM, Logistic Regression, Neural Networks
7. **Feature engineering matters** more than algorithm choice
8. **Regularize** to prevent overfitting — tune `C`, `alpha`, `max_depth`
9. **Random Forest is a great default** — robust, requires little tuning
10. **Use the test set only once** — at the very end, never during development

---

## 🛠️ Requirements

```bash
pip install numpy scipy matplotlib pandas scikit-learn
```

**Python version:** 3.7+   **scikit-learn:** 1.0+

---

## How to Run

```bash
git clone https://github.com/ridhomul/Introduction-to-Machine-Learning-with-Python.git
cd Introduction-to-Machine-Learning-with-Python
pip install numpy scipy matplotlib pandas scikit-learn
jupyter notebook
```

Open any `.ipynb` file and run cells with `Shift+Enter`.

---

## The scikit-learn API — Quick Reference

```python
# 1. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('model', SomeModel())])

# 3. Tune with CV — NEVER use X_test here
grid = GridSearchCV(pipe, {'model__param': [v1, v2]}, cv=5)
grid.fit(X_train, y_train)

# 4. Final evaluation — use test set ONCE
print(grid.score(X_test, y_test))
```

---

## 📖 About the Book

*Introduction to Machine Learning with Python* by **Andreas C. Müller** and **Sarah Guido** is one of the most popular practical ML books for Python practitioners. It emphasizes hands-on learning with scikit-learn, making it ideal for data scientists and engineers entering the ML field.

---
