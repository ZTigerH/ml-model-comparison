# ml-model-comparison

A systematic benchmark of six supervised learning algorithms across four structurally different real-world datasets, built to understand how model choice, hyperparameters, and dataset characteristics interact.

## Datasets
- **Soybean Disease Diagnosis** — categorical plant health features predicting one of four soybean diseases
- **Iris Classification** — the classic 150-row flower measurement dataset (3 species)
- **Skin Segmentation** — 245,000+ pixel samples in RGB color space, predicting skin vs. non-skin
- **Taiwanese Corporate Bankruptcy** — 6,800+ rows of financial ratios from the Taiwan Economic Journal (1999–2009), predicting company bankruptcy

## Models Evaluated
Logistic Regression, K-Nearest Neighbors (tuned across k=5–50, Euclidean & Manhattan distance), Naive Bayes, Decision Trees (tuned across max depth), Random Forest (tuned across number of estimators), and Support Vector Machines (linear & RBF kernels).

## Methodology
Each model was evaluated using 5-fold cross-validation, with features standardized where appropriate. Both predictive accuracy and runtime were tracked for every model configuration, allowing direct comparison of accuracy/speed tradeoffs across algorithm families.

## Results
On the Skin Segmentation dataset, tuned KNN and Random Forest models reached up to **96.8% cross-validated accuracy**, while Naive Bayes substantially underperformed (~24%) due to violated independence assumptions in the color-space features — a useful illustration of how model assumptions interact with real data. SVM with an RBF kernel matched the top-performing tree-based models while Decision Trees showed clear overfitting at higher depths.

## Tools
Python, scikit-learn, pandas, NumPy

## Notes
This project was completed as coursework for INFO 371 (Core Methods in Data Science) at the University of Washington.
