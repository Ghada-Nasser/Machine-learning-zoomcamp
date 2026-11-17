import pandas as pd
import numpy as np
import pickle

from pathlib import Path
from typing import Tuple

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


THRESHOLD = 0.6

numerical_columns = [
    'Administrative',
    'Administrative_Duration',
    'Informational',
    'Informational_Duration',
    'ProductRelated',
    'ProductRelated_Duration',
    'BounceRates',
    'ExitRates',
    'PageValues',
    'SpecialDay'
]

categorical_columns = [
    'Month',
    'OperatingSystems',
    'Browser',
    'Region',
    'TrafficType',
    'VisitorType',
    'Weekend'
]




def load_data() -> Tuple[pd.DataFrame, pd.Series]:

    dataset = fetch_ucirepo(id=468)

    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0]

    return X, y




def split_data(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test




def preprocessing(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, DictVectorizer]:

    dv = DictVectorizer(sparse=False)

    def prepare_features(X: pd.DataFrame) -> pd.DataFrame:

        X = X[numerical_columns + categorical_columns].copy()

        for col in categorical_columns:
            X[col] = X[col].astype(str)

        return X

    X_train = prepare_features(X_train)
    X_test = prepare_features(X_test)

    train_dict = X_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    test_dict = X_test.to_dict(orient='records')
    X_test = dv.transform(test_dict)

    return X_train, X_test, dv




def train_model(
    X_train: np.ndarray,
    y_train: pd.Series
) -> RandomForestClassifier:

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model




def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: pd.Series
) -> None:

    positive_index = list(model.classes_).index(True)

    y_proba = model.predict_proba(X_test)[:, positive_index]

    y_pred = (y_proba >= THRESHOLD)

    print("\nTest Results:")
    print(classification_report(y_test, y_pred))

    print("Threshold:", THRESHOLD)
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))




def save_model(
    model: RandomForestClassifier,
    dv: DictVectorizer
) -> None:

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    with open(models_dir / "model.pkl", "wb") as f_out:
        pickle.dump(model, f_out)

    with open(models_dir / "dv.pkl", "wb") as f_out:
        pickle.dump(dv, f_out)

    with open(models_dir / "threshold.pkl", "wb") as f_out:
        pickle.dump(THRESHOLD, f_out)

    print("\nModel, DictVectorizer and threshold saved successfully!")




def main():

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test, dv = preprocessing(X_train, X_test)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model, dv)


if __name__ == "__main__":
    main()
