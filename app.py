import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization


@st.cache_data
def load_data():
    df = pd.read_csv("athlete_events.csv")
    noc_df = pd.read_csv("noc_regions.csv")
    df = df.merge(noc_df, on="NOC", how="left")
    selected_col = ["Sex", "Age", "Height", "Weight", "region", "Sport", "Medal"]
    df = df[selected_col]
    df["Medal"] = df["Medal"].replace({np.nan: 0, "Gold": 1, "Silver": 1, "Bronze": 1})
    medal_df = df[df["Medal"] == 1]
    no_medal_df = df[df["Medal"] == 0].head(len(medal_df))
    df = pd.concat([medal_df, no_medal_df], axis=0)
    df["Height"] = df.groupby("Sex")["Height"].transform(lambda x: x.fillna(x.mean()))
    df["Weight"] = df.groupby("Sex")["Weight"].transform(lambda x: x.fillna(x.mean()))
    df["Age"] = df.groupby("Sex")["Age"].transform(lambda x: x.fillna(x.mean()))
    df.dropna(inplace=True)
    df["Height"] = df["Height"].astype(int)
    df["Weight"] = df["Weight"].astype(int)
    df["Age"] = df["Age"].astype(int)
    X = df.drop("Medal", axis=1)
    y = df["Medal"]
    return X, y


@st.cache_resource
def train_models(X, y):
    transformer = ColumnTransformer([
        ("num", StandardScaler(), ["Age", "Height", "Weight"]),
        ("cat", OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"), ["Sex", "region", "Sport"])
    ], remainder="passthrough")
    pipeline = Pipeline([
        ("transform", transformer),
        ("pca", PCA(n_components=100))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    tX_train = pipeline.fit_transform(X_train)
    tX_test = pipeline.transform(X_test)
    models = {
        "Logistic Regression": LogisticRegression(solver="sag", max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42),
        "Neural Network": Sequential([
            Dense(32, activation="relu", input_shape=[tX_train.shape[1]]),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.35),
            Dense(256, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
    }
    models["Neural Network"].compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    models["Neural Network"].fit(tX_train, np.array(y_train).reshape(-1), epochs=5,
                                 validation_data=(tX_test, np.array(y_test).reshape(-1)), verbose=0)

    trained_models = {name: model.fit(tX_train, y_train) if name != "Neural Network" else model for name, model in
                      models.items()}

    return pipeline, trained_models


def predict_medal(inputs, pipeline, models, model_choice):
    transformed_input = pipeline.transform(
        pd.DataFrame([inputs], columns=["Sex", "region", "Sport", "Height", "Weight", "Age"]))
    if model_choice == "Neural Network":
        prediction = models[model_choice].predict(transformed_input)[0][0]
        return "High" if prediction > 0.5 else "Low"
    else:
        prediction = models[model_choice].predict(transformed_input)
        return "High" if prediction[0] == 1 else "Low"


def main():
    st.sidebar.title("Olympics Data Analyzer and Predictor")
    user_menu = st.sidebar.radio(
        'Select an Option',
        (
        'Medal Tally', "Medal Predictor", 'Overall Analysis', 'Country-wise Analysis', 'Athlete-wise Analysis', "About")
    )
    X, y = load_data()
    pipeline, models = train_models(X, y)
    if user_menu == "Medal Predictor":
        st.title("Olympics Medal Predictor")
        country = X["region"].dropna().unique().tolist()
        sport = X["Sport"].dropna().unique().tolist()
        with st.form("my_form"):
            Sex = st.selectbox("Select Sex", ["M", "F"])
            Age = st.slider("Select Age", 10, 97)
            Height = st.slider("Select Height (cm)", 127, 226)
            Weight = st.slider("Select Weight (kg)", 25, 214)
            region = st.selectbox("Select Country", country)
            Sport = st.selectbox("Select Sport", sport)
            input_model = st.selectbox("Select Prediction Model", list(models.keys()))
            submitted = st.form_submit_button("Submit")
            if submitted:
                prediction = predict_medal([Sex, region, Sport, Height, Weight, Age], pipeline, models, input_model)
                with st.spinner('Predicting output...'):
                    time.sleep(1)
                    if prediction == "Low":
                        st.warning(f"Medal winning probability is {prediction}", icon="⚠️")
                    else:
                        st.success(f"Medal winning probability is {prediction}", icon="✅")


if __name__ == "__main__":
    main()
