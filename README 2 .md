# Machine Learning Assignment – Part 3: Advanced Modeling

## Project Overview

This project demonstrates advanced machine learning techniques using Scikit-learn, including Decision Trees, Random Forest, Gradient Boosting, Cross Validation, Hyperparameter Tuning, Feature Importance, Feature Ablation, Learning Curves, and Model Serialization.

---

# Dataset

* **Dataset Name:** Employee Dataset
* **Number of Records:** `<Number of rows>`
* **Number of Features:** `<Number of features>`
* **Target Variable:** Salary_Class (Created from Salary using median split)

---

# Models Implemented

* Logistic Regression
* Decision Tree (Default)
* Controlled Decision Tree
* Random Forest
* Gradient Boosting Classifier

---

# Decision Tree Overfitting Discussion

The default Decision Tree achieved very high training accuracy but lower test accuracy. This indicates overfitting because the model memorized the training data instead of learning patterns that generalize to unseen data.

Decision Trees are considered high-variance models because they make greedy splitting decisions at each node without revisiting previous splits. Small changes in the training data can produce a very different tree.

---

# Controlled Decision Tree

A second Decision Tree was trained with:

* max_depth = 5
* min_samples_split = 20

### Purpose of Hyperparameters

**max_depth**

Limits how deep the tree can grow, reducing model complexity and helping prevent overfitting.

**min_samples_split**

Prevents splitting nodes that contain very few samples, reducing splits caused by random noise.

Compared to the default tree, the controlled tree reduced the gap between training accuracy and test accuracy, indicating improved generalization.

---

# Gini vs Entropy

### Gini Impurity

```
Gini = 1 − Σ(pi²)
```

### Entropy

```
Entropy = −Σ(pi log₂(pi))
```

A Gini value of **0** means the node is perfectly pure, meaning every sample belongs to the same class.

---

# Random Forest

Random Forest combines many Decision Trees to improve prediction performance.

### Feature Importance

Random Forest computes feature importance using the average reduction in Gini impurity contributed by each feature across all trees.

Unlike Linear Regression coefficients, feature importance scores indicate the relative contribution of each feature rather than the direction of the relationship.

---

# Top 5 Important Features

| Rank | Feature       | Importance |
| ---- | ------------- | ---------: |
| 1    | `<Feature 1>` |  `<Value>` |
| 2    | `<Feature 2>` |  `<Value>` |
| 3    | `<Feature 3>` |  `<Value>` |
| 4    | `<Feature 4>` |  `<Value>` |
| 5    | `<Feature 5>` |  `<Value>` |

---

# Bagging Explanation

Random Forest uses Bagging (Bootstrap Aggregating).

Each tree is trained using a random bootstrap sample (sampling with replacement) from the training dataset.

At every split, only a random subset of features is considered.

This randomness makes individual trees less correlated, and averaging their predictions significantly reduces variance compared to a single deep Decision Tree.

---

# Feature Ablation Study

The five least important features identified by the Random Forest were removed and the model was retrained.

| Model         | Test ROC-AUC |
| ------------- | -----------: |
| Full Model    |      `<AUC>` |
| Reduced Model |      `<AUC>` |

### Interpretation

If the reduced model achieves similar or higher ROC-AUC, the removed features were largely uninformative and mainly added noise.

If ROC-AUC decreases noticeably, the removed features contributed useful information.

A simpler model can reduce inference cost and maintenance effort, but only if the performance loss remains within an acceptable threshold.

---

# Cross Validation

5-fold Stratified Cross Validation was performed using ROC-AUC.

Cross-validation provides a more reliable estimate of generalization performance because every observation is used for both training and validation across different folds, reducing dependence on a single train-test split.

| Model               |  Mean AUC |   Std AUC |
| ------------------- | --------: | --------: |
| Logistic Regression | `<Value>` | `<Value>` |
| Decision Tree       | `<Value>` | `<Value>` |
| Random Forest       | `<Value>` | `<Value>` |
| Gradient Boosting   | `<Value>` | `<Value>` |

---

# Hyperparameter Tuning

GridSearchCV was performed using the following parameter grid:

* n_estimators = [50, 100, 200]
* max_depth = [5, 10, None]
* min_samples_leaf = [1, 5]

Total parameter combinations:

```
3 × 3 × 2 = 18
```

Using 5-fold cross-validation:

```
18 × 5 = 90 model fits
```

### Best Parameters

```
<grid.best_params_>
```

### Best Cross-Validation Score

```
<grid.best_score_>
```

Grid Search evaluates every possible parameter combination and usually finds the optimal configuration but requires more computation than Randomized Search, which evaluates only a subset of combinations.

---

# Manual Learning Curve

| Training Fraction | Training AUC |  Test AUC |
| ----------------: | -----------: | --------: |
|               20% |    `<Value>` | `<Value>` |
|               40% |    `<Value>` | `<Value>` |
|               60% |    `<Value>` | `<Value>` |
|               80% |    `<Value>` | `<Value>` |
|              100% |    `<Value>` | `<Value>` |

### Interpretation

* Training AUC generally decreases as more training data becomes available because the model has fewer opportunities to overfit small datasets.
* Test AUC generally increases with additional training data, indicating improved generalization.
* If Test AUC continues increasing at 100% of the available data, the model is likely data-limited.
* If Test AUC has plateaued, the model is likely limited by its capacity rather than the amount of data.

---

# Model Serialization

The best-performing pipeline was saved using Joblib.

```
best_model.pkl
```

The saved model was successfully reloaded and used to predict new samples without errors.

---

# Final Model Comparison

| Model               | 5-Fold Mean AUC | 5-Fold Std AUC |  Test AUC |
| ------------------- | --------------: | -------------: | --------: |
| Logistic Regression |       `<Value>` |      `<Value>` | `<Value>` |
| Decision Tree       |       `<Value>` |      `<Value>` | `<Value>` |
| Random Forest       |       `<Value>` |      `<Value>` | `<Value>` |
| Gradient Boosting   |       `<Value>` |      `<Value>` | `<Value>` |

---

# Final Recommendation

Based on the cross-validation results and test-set ROC-AUC, **<Best Model Name>** is recommended for deployment.

This model achieved the highest predictive performance while maintaining stable cross-validation scores. It demonstrated strong generalization capability, reduced overfitting compared to a single Decision Tree, and provided reliable performance on unseen data. Therefore, it offers the best balance between accuracy, robustness, and practical deployment.
