# IBM HR Analytics — Employee Attrition & Performance
**Project 1**

End-to-end ML pipeline to predict employee attrition using the IBM HR Analytics dataset from Kaggle.

## What's covered
- EDA — distributions, correlation heatmap, pairplot, boxplots, stacked bar charts, feature correlation with target
- Preprocessing — dropped constant/ID columns, label encoded target, one-hot encoded categoricals
- Dimensionality Reduction — PCA (for visualization)
- Modeling — Logistic Regression, Random Forest, SVM, KNN, XGBoost
- Tuning — GridSearchCV with ROC-AUC scoring
- Evaluation — Confusion matrices, ROC curves
- Interpretability — Random Forest feature importance, SHAP summary plot

## Best model
SVM (linear kernel, C=0.1) — ROC-AUC: 0.811

---

## Future improvements

- [ ] **Class imbalance handling** — Apply SMOTE or `class_weight='balanced'` to address the 84/16 split. Currently models are biased toward predicting "Stayed."
- [ ] **Feature engineering** — Add derived features: income per year of experience, satisfaction composite score, tenure per role, promotion stagnation ratio.
- [ ] **Focused EDA with commentary** — Reduce to 5-6 most insightful findings with markdown explanations (e.g. "Employees who work overtime show significantly higher attrition — indicating burnout").
- [ ] **Connect PCA to modeling** — Either use PCA-reduced features in at least one model for comparison, or remove the section.
- [ ] **Cross-validation scores** — Add `cross_val_score` with 5-fold CV to report variance in model estimates beyond a single train-test split.
- [ ] **Business narrative** — Add intro explaining the business problem and a conclusion section summarizing top findings and recommended HR actions.
