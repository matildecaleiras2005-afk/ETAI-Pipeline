import os
import numpy as np
from src.evaluate import corrected_paired_test

def save_test_result(results_dir: str, model_name: str, result: dict):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "statistical_tests.txt")
    
    report = (
        f"Model Comparison: {model_name}\n"
        f"Mean Difference: {result['mean_diff']:.4f}\n"
        f"Wins (With vs Without): {result['wins']}\n"
        f"t-statistic: {result['t']:.4f}\n"
        f"p-value: {result['p_value']:.4f}\n"
        + "-" * 40 + "\n"
    )
    
    with open(path, "a") as f:
        f.write(report)
    print(f"\n[Saved] Result appended to {path}")

# --- Dados ---
scores_without_selection = [0.722, 0.704, 0.735, 0.737, 0.739]
scores_with_selection    = [0.717, 0.705, 0.729, 0.732, 0.736]

# 5 stratified folds: test_train_ratio = 1 / (n_splits - 1)
n_splits = 5
test_train_ratio = 1.0 / (n_splits - 1)

result = corrected_paired_test(scores_without_selection, scores_with_selection, test_train_ratio)

print(" --- Corrected Paired t-Test ---")
print(f"Mean Difference: {result['mean_diff']:.4f}")
print(f"Wins (With vs Without): {result['wins']}")
print(f"t-statistic: {result['t']:.4f}")
print(f"p-value: {result['p_value']:.4f}")

model_name = "HistGradientBoosting - Feature Selection vs Off"  # Replace with the actual model name if needed
save_test_result("results", model_name, result)