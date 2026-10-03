"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> preprocess -> split -> train
    -> evaluate (train & test) -> save results
"""
from sklearn.model_selection import train_test_split
import yaml

from src.data import load_data
from src.preprocessing import (
    clean_dataset, drop_duplicate_rows, split_features_target, build_preprocessor, split_dev_test,
)
from src.model import build_model
from src.evaluate import (
    fairness_report, evaluate
)
from src.results import save_run
from sklearn.pipeline import Pipeline


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    
    df = load_data(config["data"]["path"])
    df_clean = clean_dataset(df, config["diagnostics"])
    df_clean = drop_duplicate_rows(df_clean, id_column=config["data"].get("id_column"))

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # week 4: the final test set is set aside HERE and never used again in this script.
    # Every decision from now on (preprocessing, model, hyperparameters) is made on the
    # development set only. The test set is used only for the final assessment.
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    X_tr, X_va, y_tr, y_va, extras_tr, extras_va = train_test_split(
        X_dev, y_dev, extras_dev, 
        test_size=0.25, 
        random_state=42, 
        stratify=y_dev
    )

    # training part of every fold -- the validation fold never leaks into its own preprocessing
    pipeline = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])

    pipeline.fit(X_tr, y_tr)

    # predict on both splits -- train accuracy vs. test accuracy is how we'll spot overfitting, not just how "good" the model looks
    y_train_pred = pipeline.predict(X_tr)
    y_va_pred = pipeline.predict(X_va)
    report = evaluate(y_tr, y_train_pred, y_va, y_va_pred)

    report += "\n" + fairness_report(
        y_va, y_va_pred, extras_va, sensitive_attr=config["data"]["sensitive_attr"]
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()

