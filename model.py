import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from all_imports import BASE_DIR, CSV_PATH


def load_data() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def preprocess(df: pd.DataFrame):
    df = df.drop(columns=[c for c in ["ca", "id"] if c in df.columns])

    categorical_cols = ["sex", "dataset", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    numerical_cols = df.drop(columns=["num"] + categorical_cols, errors="ignore").columns

    # Missing values: median for numerical, mode for categorical
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop(columns=["num"])
    # Collapse multi-class severity (1–4) into binary: 0 = no disease, 1 = disease
    y = (df["num"] > 0).astype(int)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train(X_train, y_train) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf: RandomForestClassifier, X_test, y_test, feature_names) -> None:
    y_pred = clf.predict(X_test)

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    out_path = BASE_DIR / "images" / "confmatx.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {out_path}")

    importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nTop 10 Features:\n", importances.head(10))


def save_artifacts(clf: RandomForestClassifier, columns) -> None:
    joblib.dump(clf, BASE_DIR / "heart_disease_model.pkl")
    joblib.dump(columns, BASE_DIR / "model_columns.pkl")
    print("Saved: heart_disease_model.pkl, model_columns.pkl")


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print("Target distribution:\n", y_train.value_counts(normalize=True))

    clf = train(X_train, y_train)
    evaluate(clf, X_test, y_test, X_train.columns)
    save_artifacts(clf, X_train.columns)
