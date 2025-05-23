# 🏡 House Price Prediction - Streamlit App

A Streamlit web app that allows users to explore, train, and compare regression models (Linear Regression vs Random Forest) on the Ames Housing Dataset.

## 🚀 Features

- Data cleaning & preprocessing
- Feature engineering (TotalSF, HouseAge, etc.)
- Encoding categorical variables
- Scaling numerical features
- Cross-validation with R² scores
- Model performance comparison on Train & Test sets
- Visualizations:
  - SalePrice distribution
  - Outlier detection via boxplot
  - Feature importance from Random Forest

## 📁 Dataset

The app uses the **Ames Housing dataset**, which must be available as `AmesHousing.csv` in the project folder.

You can download it from:
- [UCI Machine Learning Repo](https://archive.ics.uci.edu/ml/datasets/Ames+Housing)
- Or use your own regression dataset (numeric target)

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
````

**Or for conda:**

```bash
conda create -n house-price python=3.10
conda activate house-price
pip install -r requirements.txt
```

## 🛠️ Models Used

* **Linear Regression** (baseline)
* **Random Forest Regressor**

## 📊 Performance Metrics

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score

---

## 📌 Notes

* SHAP explainability was removed due to dependency issues during deployment.
* This app is optimized for regression datasets only.

---

