# ml-model-comparison

A systematic benchmark of six supervised learning algorithms across four structurally different real-world datasets, built to understand how model choice, hyperparameters, and dataset characteristics interact.

| Dataset | Rows | Task | Majority-class baseline |
|---|---|---|---|
| Soybean Disease | 47 | 4-class disease diagnosis from categorical features | 36% |
| Iris | 150 | 3-class species classification | 33% |
| Skin Segmentation | 245,057 | Skin vs. non-skin from RGB pixel values | 79% |
| Taiwanese Bankruptcy | 6,819 | Bankrupt vs. not, from 95 financial ratios (1999-2009) | 97% |


## Models Evaluated
Logistic Regression, K-Nearest Neighbors (tuned across k=5–50, Euclidean & Manhattan distance), Naive Bayes, Decision Trees (tuned across max depth), Random Forest (tuned across number of estimators), and Support Vector Machines (linear & RBF kernels).

## Methodology
- Models: Logistic Regression, K-Nearest Neighbors (k = 5 to 50, Euclidean and Manhattan), Naive Bayes, Decision Trees (max depth 5 to 25), Random Forest (10, 50, 100 trees), SVM (linear and RBF)
- 5-fold cross-validation, reporting mean accuracy and wall-clock runtime
- Features standardized for Logistic Regression, KNN, and SVM

## Results
**Skin Segmentation (large, low-dimensional).** KNN was the most accurate model at about 99.2% CV accuracy, well above the 79% baseline. Tuned SVM (RBF) reached 98.3% but took about 88 seconds, versus about 1 second for KNN. Linear SVM was both slower (about 292 seconds) and much less accurate (91.9%), which shows the classes aren't linearly separable. Random Forest and Decision Trees landed near 98.1%.

**Bankruptcy (large, high-dimensional, heavily imbalanced).** Only 3.2% of companies are bankrupt, so predicting "not bankrupt" every time scores 96.8%. The best models (KNN, Random Forest, RBF SVM) reached about 96.8%, which only matches that baseline. Accuracy is the wrong metric for this problem, so these results show models failing to beat the baseline, not strong performance. Naive Bayes scored 24%, far below baseline. Deeper decision trees (about 94%) did worse than a depth-5 tree (96.2%), a sign of overfitting.

**Small datasets (Soybean, Iris).** With 47 and 150 rows, many models hit 96 to 100%, so differences are mostly noise. On Soybean, KNN accuracy falls to the baseline once k is large relative to the data, because each neighborhood spans multiple classes.

## Limitations

- Accuracy only. A follow-up should add precision, recall, F1, or AUC for Bankruptcy, with class weighting.
- Hyperparameters were compared over a manual grid, not a nested search.
- The scaler was fit on the full dataset before cross-validation, which causes minor leakage.
- No random seeds were set, so exact numbers vary slightly between runs.


## Tools
Python, scikit-learn, pandas, NumPy

## Notes
This project was completed as coursework for INFO 371 (Core Methods in Data Science) at the University of Washington.
