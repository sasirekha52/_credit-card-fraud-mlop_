import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from evaluate import evaluate
from preprocess import FEATURE_COLUMNS, TARGET_COLUMN

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")

X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
X_val, y_val = val[FEATURE_COLUMNS], val[TARGET_COLUMN]


def make_pipeline(strategy: str) -> Pipeline:
    steps = []
    if strategy == "smote":
        steps.append(("smote", SMOTE(sampling_strategy=0.1, random_state=42)))
    class_weight = "balanced_subsample" if strategy == "class_weight" else None
    steps.append(
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=5,
                class_weight=class_weight,
                n_jobs=-1,
                random_state=42,
            ),
        )
    )
    return Pipeline(steps)


rows = []
for strategy in ["none", "class_weight", "smote"]:
    print(f"Training Random Forest with strategy: {strategy} ...")
    pipe = make_pipeline(strategy)
    pipe.fit(X_train, y_train)
    scores = pipe.predict_proba(X_val)[:, 1]
    rows.append({"strategy": strategy, **evaluate(y_val, scores)})

results = pd.DataFrame(rows).set_index("strategy")
pd.set_option("display.width", 200)
print("\nRandom Forest on VALIDATION set (threshold 0.5)")
print(results.round(4))