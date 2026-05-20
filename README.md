# IBM HR Analytics — Employee Attrition & Performance

Dataset: IBM HR Analytics (1,470 employees, 35 columns) from Kaggle.

---

## Project 1 — Attrition Prediction

End-to-end ML pipeline to predict employee attrition.

**Notebook:** [IBM_HR_Attrition.ipynb](IBM_HR_Attrition.ipynb)

### What's covered
- EDA — distributions, correlation heatmap, pairplot, boxplots, stacked bar charts, feature correlation with target
- Preprocessing — dropped constant/ID columns, label encoded target, one-hot encoded categoricals
- Dimensionality Reduction — PCA (for visualization)
- Modeling — Logistic Regression, Random Forest, SVM, KNN, XGBoost
- Tuning — GridSearchCV with ROC-AUC scoring
- Evaluation — Confusion matrices, ROC curves
- Interpretability — Random Forest feature importance, SHAP summary plot

### Best model
SVM (linear kernel, C=0.1) — ROC-AUC: 0.811

### Future improvements
- [ ] **Class imbalance handling** — Apply SMOTE or `class_weight='balanced'` to address the 84/16 split. Currently models are biased toward predicting "Stayed."
- [ ] **Feature engineering** — Add derived features: income per year of experience, satisfaction composite score, tenure per role, promotion stagnation ratio.
- [ ] **Focused EDA with commentary** — Reduce to 5-6 most insightful findings with markdown explanations (e.g. "Employees who work overtime show significantly higher attrition — indicating burnout").
- [ ] **Connect PCA to modeling** — Either use PCA-reduced features in at least one model for comparison, or remove the section.
- [ ] **Cross-validation scores** — Add `cross_val_score` with 5-fold CV to report variance in model estimates beyond a single train-test split.
- [ ] **Business narrative** — Add intro explaining the business problem and a conclusion section summarizing top findings and recommended HR actions.

---

## Project 2 — Attrition Pattern Mining

Groupby-based pattern mining to find employee segments with high attrition rates.

**Script:** [Project2_AssociationRules/ibm_hr_association_rules.py](Project2_AssociationRules/ibm_hr_association_rules.py)

**Output:** `Project2_AssociationRules/attrition_groupby_rules.csv`

### Approach
- Bins all numerical columns into labeled ranges (e.g. Age → Young/Mid/Senior)
- Maps ordinal scales (satisfaction, job level, education) to readable labels
- Groups employees by every 1-, 2-, and 3-column combination (27 feature columns, ~87K groups)
- For each group: counts Attrition Yes/No, computes attrition rate and lift vs. overall population (~16.1%)
- Filters groups with fewer than 10 employees

### Key findings
- **Sales Representatives** have the highest single-feature attrition rate (40%, 2.47x baseline)
- **OverTime=Yes** affects 28% of employees and drives 31% attrition (1.89x)
- **Low Job Involvement + Low Environment Satisfaction** combinations push attrition above 80–90%
- Senior employees (JobLevel Senior/Executive, long tenure) have near-zero attrition (<5%)
- Strongest 3-column signal: `EnvironmentSatisfaction=Low + MonthlyIncome=Low + OverTime=Yes` → 88% attrition (15/17 employees)
