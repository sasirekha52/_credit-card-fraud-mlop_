import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

from evaluate import evaluate
from preprocess import FEATURE_COLUMNS, TARGET_COLUMN

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")

X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
X_val, y_val = val[FEATURE_COLUMNS], val[TARGET_COLUMN]

# legitimate count / fraud count, computed from TRAINING data only
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
SCALE_POS_WEIGHT = neg / pos
print(f"scale_pos_weight = {SCALE_POS_WEIGHT:.1f}  (legit={neg}, fraud={pos})\n")


def make_pipeline(strategy: str) -> Pipeline:
    steps = []
    if strategy == "smote":
        steps.append(("smote", SMOTE(sampling_strategy=0.1, random_state=42)))
    weight = SCALE_POS_WEIGHT if strategy == "class_weight" else 1.0
    steps.append(
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=weight,
                tree_method="hist",
                eval_metric="aucpr",
                n_jobs=-1,
                random_state=42,
            ),
        )
    )
    return Pipeline(steps)


rows = []
for strategy in ["none", "class_weight", "smote"]:
    print(f"Training XGBoost with strategy: {strategy} ...")
    pipe = make_pipeline(strategy)
    pipe.fit(X_train, y_train)
    scores = pipe.predict_proba(X_val)[:, 1]
    rows.append({"strategy": strategy, **evaluate(y_val, scores)})

results = pd.DataFrame(rows).set_index("strategy")
pd.set_option("display.width", 200)
print("\nXGBoost on VALIDATION set (threshold 0.5)")
print(results.round(4))