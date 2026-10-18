import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluate import evaluate
from preprocess import FEATURE_COLUMNS, TARGET_COLUMN

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")

X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
X_val, y_val = val[FEATURE_COLUMNS], val[TARGET_COLUMN]


def make_pipeline(strategy: str) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if strategy == "smote":
        # fraud becomes 10% of the size of the legitimate class, in training only
        steps.append(("smote", SMOTE(sampling_strategy=0.1, random_state=42)))
    class_weight = "balanced" if strategy == "class_weight" else None
    steps.append(
        ("model", LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42))
    )
    return Pipeline(steps)


rows = []
for strategy in ["none", "class_weight", "smote"]:
    pipe = make_pipeline(strategy)
    pipe.fit(X_train, y_train)
    scores = pipe.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val, scores)
    rows.append({"strategy": strategy, **metrics})

results = pd.DataFrame(rows).set_index("strategy")
pd.set_option("display.width", 200)
print("Logistic Regression on VALIDATION set (threshold 0.5)")
print(results.round(4))