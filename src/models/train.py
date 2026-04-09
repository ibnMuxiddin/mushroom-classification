import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score

import time

from src.data.load import load_raw_data
from src.data.preprocess import clean_data


def main():

    print("Salom main ishladi", flush=True)
    # Load data
    df = load_raw_data("mushrooms.csv")

    # Clean data
    df = clean_data(df)

    # Split target column
    X = df.drop('class', axis=1)
    y = df['class']

    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23, stratify=y)

    # Split columns cat and numeric
    cat_cols = X.select_dtypes(include='str').columns
    num_cols = X.select_dtypes(exclude='str').columns

    # Preprocessing pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='median'))
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ColumnTransformer
    preprocesser = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )

    # list of models
    models = {
        "logreg" : LogisticRegression(),
        "k-nn": KNeighborsClassifier(),
        "randfor": RandomForestClassifier(),
        "dectree": DecisionTreeClassifier()
    }

    # train models

    result = []

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessing", preprocesser),
            ("model", model)
        ])

        start_tr = time.time()
        pipeline.fit(X_train, y_train)
        end_tr = time.time()

        start_pr = time.time()
        y_predict = pipeline.predict(X_test)
        end_pr = time.time()

        accur = accuracy_score(y_test, y_predict)
        f1_sco = f1_score(y_test, y_predict, average="weighted")
        prec = precision_score(y_test, y_predict, average="weighted")
        train_time = end_tr - start_tr
        predict_time = end_pr-start_pr

        result.append({
            "name": name,
            "accur": accur,
            "f1-score": f1_sco,
            "precision": prec,
            "train_time": train_time,
            "predict_time": predict_time
        })

    # Show results
    result_df = pd.DataFrame(result)
    print(result_df)


# Run teh main() function
if __name__ == "__main__":
    main()


