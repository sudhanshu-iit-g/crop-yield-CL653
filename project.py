import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up the page
st.set_page_config(page_title="Crop Yield Predictor", layout="wide")
st.title("🌾 Crop Yield Prediction Dashboard")

# File uploader
st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using default dataset (yield_df.csv)")
    df = pd.read_csv("yield_df.csv")

# Data Preprocessing
df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
df.drop_duplicates(inplace=True)
df["average_rain_fall_mm_per_year"] = df["average_rain_fall_mm_per_year"].astype(np.float64)

# Sidebar navigation
option = st.sidebar.radio("Choose an option:", ["EDA", "Model Training", "Model Comparison", "Predict"])

if option == "EDA":
    st.subheader("Exploratory Data Analysis")

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Statistical Summary")
    st.dataframe(df.describe())

    st.write("### Column Data Types")
    st.dataframe(df.dtypes)

    st.write("### Missing Values")
    st.dataframe(df.isnull().sum())

    st.write("### Target Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["hg/ha_yield"], kde=True, ax=ax, color="green")
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("### Boxplot for Outliers")
    num_col = st.selectbox("Select column to view boxplot", df.select_dtypes(include=np.number).columns)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=num_col, ax=ax)
    st.pyplot(fig)

    st.write("### Pairplot of Numerical Features")
    if st.checkbox("Show Pairplot"):
        fig = sns.pairplot(df.select_dtypes(include=np.number))
        st.pyplot(fig)

    st.write("### Distribution of Target vs Categorical Features")
    cat_col = st.selectbox("Select categorical column", df.select_dtypes(include='object').columns)
    fig, ax = plt.subplots()
    sns.boxplot(x=df[cat_col], y=df["hg/ha_yield"], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

elif option == "Model Training":
    st.subheader("Model Training")
    df_encoded = pd.get_dummies(df.drop("hg/ha_yield", axis=1))
    X = df_encoded
    y = df["hg/ha_yield"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest", "Gradient Boosting", "SVR", "Decision Tree"])

    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Random Forest":
        model = RandomForestRegressor()
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor()
    elif model_choice == "SVR":
        model = SVR()
    else:
        model = DecisionTreeRegressor()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("### Evaluation Metrics")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

    if hasattr(model, 'feature_importances_'):
        st.write("### Feature Importances")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', ax=ax, palette='mako')
        ax.set_title("Top 10 Important Features")
        st.pyplot(fig)

    st.session_state.model = model
    st.session_state.features = X.columns.tolist()
    st.session_state.original_columns = df.drop("hg/ha_yield", axis=1).columns.tolist()

elif option == "Model Comparison":
    st.subheader("Model Comparison")
    df_encoded = pd.get_dummies(df.drop("hg/ha_yield", axis=1))
    X = df_encoded
    y = df["hg/ha_yield"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "SVR": SVR(),
        "Decision Tree": DecisionTreeRegressor()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        })

    results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
    st.dataframe(results_df)

    st.write("### R2 Score Comparison")
    fig, ax = plt.subplots()
    sns.barplot(data=results_df, x="Model", y="R2", ax=ax, palette="viridis")
    ax.set_title("Model R2 Score Comparison")
    st.pyplot(fig)

    st.write("### MAE Comparison")
    fig, ax = plt.subplots()
    sns.barplot(data=results_df, x="Model", y="MAE", ax=ax, palette="crest")
    ax.set_title("Model MAE Comparison")
    st.pyplot(fig)

    st.write("### MSE Comparison")
    fig, ax = plt.subplots()
    sns.barplot(data=results_df, x="Model", y="MSE", ax=ax, palette="flare")
    ax.set_title("Model MSE Comparison")
    st.pyplot(fig)

elif option == "Predict":
    st.subheader("Make a Prediction")
    st.write("Use the trained model to make predictions on new data")

    if "model" not in st.session_state or "features" not in st.session_state:
        st.warning("Please train a model first in the 'Model Training' tab.")
    else:
        st.markdown("<h3 style='text-align: center; color: red;'>Input All Features Here</h3>", unsafe_allow_html=True)
        input_data = {}

        for col in st.session_state.original_columns:
            if df[col].dtype == object:
                input_data[col] = st.selectbox(col, options=df[col].dropna().unique())
            elif df[col].dtype in [int, np.int64]:
                input_data[col] = st.number_input(col, value=int(df[col].median()))
            else:
                input_data[col] = st.number_input(col, value=float(df[col].median()))

        if st.button("Predict", type="primary"):
            input_df = pd.DataFrame([input_data])
            input_encoded = pd.get_dummies(input_df)
            model_features = st.session_state.features
            for col in model_features:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[model_features]

            prediction = st.session_state.model.predict(input_encoded)[0]
            st.success(f"Predicted Yield: {prediction:.2f} hg/ha")
