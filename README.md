
# Titanic Kaggle Competition - Survival Prediction

This repository contains the code and models used to predict the survival of passengers aboard the Titanic using machine learning algorithms. The project includes data preprocessing, feature engineering, and model training, employing various models like Logistic Regression, Random Forest, and XGBoost.

## Table of Contents

- [Overview](#overview)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Results](#results)
- [Submission](#submission)
- [Dependencies](#dependencies)

## Overview

This project is part of the Kaggle Titanic competition, where the objective is to predict whether a passenger survived or not. The dataset includes features such as age, sex, class, and embarkation port. We used several machine learning models to classify survival.

## Data Preprocessing

1. **Handling Missing Values:**  
   - Age values are missing for some passengers. We impute missing values based on the mean age of the passengers in each class (Pclass).
   - Categorical variables like 'Sex' and 'Embarked' are encoded using one-hot encoding.

2. **Feature Engineering:**  
   - 'Fare' values are transformed using the natural logarithm to reduce the skewness.
   - One-hot encoding is applied to 'Sex' and 'Embarked' columns to convert them into numerical features.
   - Rows with missing values are dropped, ensuring a clean training dataset.

3. **Feature Selection:**  
   The following features were used for modeling:
   - `Pclass`, `Age`, `Sex`, `Fare`, `Embarked`

## Modeling

### Logistic Regression

We trained a Logistic Regression model with the following steps:
- Used `Pclass` to impute missing `Age` values.
- Encoded categorical variables like `Sex` and `Embarked`.
- Applied the model to the training set and made predictions on the test set.

```python
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Random Forest

We also employed Random Forest for comparison:

```python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
```

### XGBoost

Finally, we used XGBoost, a gradient boosting method:

```python
import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
```

## Results

After training and evaluating the models, we obtained the following results:

### Logistic Regression
- **Accuracy:** 93.3%
- **Classification Report:**  
  Precision: 0.87 (Survived), 0.97 (Did not survive)  
  Recall: 0.95 (Survived), 0.92 (Did not survive)  
  F1-Score: 0.91 (Survived), 0.95 (Did not survive)

### Random Forest
- **Accuracy:** 91.2%
- Random Forest performed well but with slightly lower accuracy than Logistic Regression.

### XGBoost
- **Accuracy:** 94.1%
- XGBoost showed the best performance among the models, making it the final choice for submission.

### Confusion Matrix

```plaintext
[[245  21]  
 [  7 145]]
```

### Mean Squared Error:
- **Logistic Regression:** 0.067
- **Random Forest:** 0.085
- **XGBoost:** 0.052

## Submission

After making predictions using the best-performing model (XGBoost), we generated the final submission file:

```python
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': predictions.astype(int)
})
submission.to_csv('submission.csv', index=False)
```

The submission file is ready to be uploaded to Kaggle for evaluation.

## Dependencies

To run this project, you'll need the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`

You can install them using pip:

```bash
pip install pandas numpy scikit-learn xgboost
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
