import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Judul aplikasi
st.title("Ames Housing Price Prediction")

# Upload file
uploaded_file = st.file_uploader("Upload dataset AmesHousing.csv", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview Data")
    st.dataframe(df.head())

    # Data cleaning
    df_clean = df.copy()
    df_clean.drop(columns=['Order', 'PID'], inplace=True, errors='ignore')
    
    # Drop kolom dengan >80% missing
    missing = df_clean.isnull().sum()
    to_drop = missing[missing > 0.8 * len(df_clean)].index.tolist()
    df_clean.drop(columns=to_drop, inplace=True)

    # Isi missing value
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype == "object":
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Fitur baru
    df_clean['TotalSF'] = df_clean['Total Bsmt SF'] + df_clean['1st Flr SF'] + df_clean['2nd Flr SF']
    df_clean['HouseAge'] = df_clean['Yr Sold'] - df_clean['Year Built']
    df_clean['RemodAge'] = df_clean['Yr Sold'] - df_clean['Year Remod/Add']

    # Encoding
    label_cols = [col for col in df_clean.select_dtypes(include='object').columns if df_clean[col].nunique() <= 10]
    for col in label_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])

    df_model = pd.get_dummies(df_clean, drop_first=True)

    # Split data
    X = df_model.drop(columns=['SalePrice'])
    y = df_model['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    lr = LinearRegression()
    rf = RandomForestRegressor(random_state=42)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_lr = cross_val_score(lr, X_train_scaled, y_train, cv=cv, scoring='r2')
    cv_scores_rf = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring='r2')

    st.subheader("Cross-Validation Scores")
    st.write(f"Linear Regression Mean R²: {np.mean(cv_scores_lr):.4f}")
    st.write(f"Random Forest Mean R²: {np.mean(cv_scores_rf):.4f}")

    # Train final model
    lr.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)

    def calculate_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, rmse, r2

    metrics = {
        'Model': [], 'MAE Train': [], 'MSE Train': [], 'RMSE Train': [], 'R² Train': [],
        'MAE Test': [], 'MSE Test': [], 'RMSE Test': [], 'R² Test': []
    }

    for name, model in zip(["Linear Regression", "Random Forest"], [lr, rf]):
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)
        metrics['Model'].append(name)
        metrics['MAE Train'].append(train_metrics[0])
        metrics['MSE Train'].append(train_metrics[1])
        metrics['RMSE Train'].append(train_metrics[2])
        metrics['R² Train'].append(train_metrics[3])
        metrics['MAE Test'].append(test_metrics[0])
        metrics['MSE Test'].append(test_metrics[1])
        metrics['RMSE Test'].append(test_metrics[2])
        metrics['R² Test'].append(test_metrics[3])

    st.subheader("Model Performance")
    st.dataframe(pd.DataFrame(metrics))

    # Visualisasi distribusi
    st.subheader("Distribusi SalePrice")
    fig1, ax1 = plt.subplots()
    sns.histplot(y, kde=True, ax=ax1)
    st.pyplot(fig1)

    # Boxplot
    st.subheader("Boxplot SalePrice")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=y, ax=ax2)
    st.pyplot(fig2)

    # Feature Importance Random Forest
    st.subheader("Top 20 Fitur Terpenting (Random Forest)")
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(20)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index, ax=ax3)
    ax3.set_title("Feature Importance")
    st.pyplot(fig3)
