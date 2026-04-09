# Data bilan ishlash
import pandas as pd

# Malumotlarni ajratish uchun
from sklearn.model_selection import train_test_split, GridSearchCV

# Pipeline 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Malumotga ishlov berish uchun
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Data load and clean
from src.data.load import load_raw_data
from src.data.preprocess import clean_data

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Modelni baholash
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Path 
from pathlib import Path

# save model
import joblib


def main():
    
    # load data
    df = load_raw_data("mushrooms.csv")

    # clean data
    df = clean_data(df)

    # slit target column
    X = df.drop('class', axis=1)
    y = df['class']

    # split train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23, stratify=y)

    # split columns numeric and categorical columns
    num_cols = X.select_dtypes(exclude="str").columns
    cat_cols = X.select_dtypes(include="str").columns

    # pipeline for numeric data
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # pipeline for categorical data
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    # ColumnTranformer
    preprocesser = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )

    # Final pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocesser),
        ("model", DecisionTreeClassifier())
    ])

    # param_grid
    # model__ -- pipeline ichidagi modelga murojaat
    param_grid = {
        "model__max_depth": [3, 5, 10, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    }

    # GridSearchCV
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )

    # Train model
    grid.fit(X_train, y_train)

    # The best model
    best_model = grid.best_estimator_

    y_predict = best_model.predict(X_test)

    print(classification_report(y_test, y_predict))
    print(confusion_matrix(y_test, y_predict))


    joblib.dump(best_model, "models/best_model.pkl")

    # Show results
    #print("Best params:", grid.best_params_)
    #print("Best score:", grid.best_score_)

if __name__ == "__main__":
    main()






