# Model Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression achieved the highest precision, 0.7366, meaning that when it predicts a purchase, it is often correct. However, recall is only 0.3586, so it misses many actual purchase sessions. |
| Decision Tree | Decision Tree achieved the highest accuracy, AUC, and MCC. Its AUC of 0.9037 and MCC of 0.5718 show strong ranking and balanced classification performance on this imbalanced dataset. |
| K-Nearest Neighbors | KNN produced reasonable accuracy but weak recall, 0.3246. This suggests that similarity-based prediction struggled to identify many positive purchase sessions after preprocessing. |
| Naive Bayes | Naive Bayes had high recall, 0.6702, but low precision, 0.3257. It identifies many purchase sessions but creates many false positives, which fits its simplifying independence assumption. |
| Support Vector Machine | SVM achieved the best recall and F1 score. It is the strongest model for finding actual purchase sessions, although its precision is lower than Logistic Regression and Decision Tree. |
| Overall Winner for your dataset? | Decision Tree is the overall winner because it leads Accuracy, AUC, and MCC while maintaining a strong F1 score. SVM is a close alternative when recall is the highest priority, but Decision Tree is more balanced across the full metric set. |
