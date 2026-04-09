# mushroom-classification
# 🍄 Mushroom Classification Project

## 📌 Project Overview

This project aims to classify mushrooms as **edible** or **poisonous** using machine learning techniques.
The goal is to build a reliable and production-ready ML pipeline that can handle real-world data.

---

## 🚀 Project Workflow

The project follows a structured Data Science workflow:

1. **Data Loading**
2. **Data Cleaning**
3. **Feature Engineering**
4. **Model Training**
5. **Model Evaluation**
6. **Hyperparameter Tuning**
7. **Model Saving**
8. **Prediction Pipeline**

---

## 🗂️ Project Structure

```
mushroom-classification/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│
├── src/
│   ├── data/
│   │   ├── load.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── tune.py
│   │   └── predict.py
│
├── models/
│   └── best_model.pkl
│
└── README.md
```

---

## ⚙️ Pipeline Design

The project uses a **Scikit-learn Pipeline** to ensure:

* No data leakage
* Reproducibility
* Clean and modular code

Pipeline includes:

* Missing value imputation
* One-Hot Encoding (`handle_unknown="ignore"`)
* Decision Tree Classifier (tuned)

---

## 🤖 Model Selection

Multiple models were evaluated:

* Logistic Regression
* Random Forest
* K-Nearest Neighbors
* Decision Tree

After comparing:

* Accuracy
* F1-score
* Training time
* Prediction time

👉 **Decision Tree** was selected as the best model.

---

## 🔧 Hyperparameter Tuning

Hyperparameter tuning was performed using GridSearchCV with 5-fold cross-validation.

Optimized parameters include:

* `max_depth`
* `min_samples_split`
* `min_samples_leaf`

---

## 📊 Evaluation

Final evaluation was done on a **held-out test set**.

Metrics used:

* Accuracy
* F1-score (weighted)

---

## 💾 Model Saving

The trained pipeline (preprocessing + model) is saved using `joblib`:

```
models/best_model.pkl
```

---

## 🔮 Prediction

To make predictions on new data:

```
python -m src.models.predict
```

The model automatically:

* Cleans data
* Applies feature engineering
* Encodes categorical variables
* Makes predictions

---

## 🧠 Key Learnings

* Importance of avoiding data leakage
* Proper use of train/test split
* Pipeline-based ML design
* Difference between train time and prediction time
* Handling unseen categories in production

---

## 📌 Future Improvements

* Move feature engineering into pipeline
* Add model versioning
* Build API using FastAPI
* Deploy model to cloud

---

## 👤 Author

Azizbek Sunnat

