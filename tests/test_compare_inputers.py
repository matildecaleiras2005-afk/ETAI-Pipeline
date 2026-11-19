import os
from src.evaluate import corrected_paired_test

def save_test_result(results_dir: str, comparison_name: str, result: dict) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "statistical_tests.txt")

    report = (
        f"Imputer Comparison: {comparison_name}\n"
        f"Mean Difference: {result['mean_diff']:.4f}\n"
        f"Wins: {result['wins']}\n"
        f"t-statistic: {result['t']:.4f}\n"
        f"p-value: {result['p_value']:.4f}\n"
        + "-" * 40
        + "\n"
    )

    with open(path, "a", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Saved] Result appended to {path}")


scores_median = [0.715, 0.705, 0.729, 0.730, 0.735]
scores_knn = [0.717, 0.706, 0.726, 0.729, 0.732]
scores_iterative = [0.718, 0.705, 0.726, 0.728, 0.733]
scores_random_forest = [0.714, 0.703, 0.719, 0.727, 0.735]

n_splits = 5
test_train_ratio = 1.0 / (n_splits - 1)

comparisons = [
    ("KNN vs Median", scores_knn, scores_median),
    ("Iterative vs Median", scores_iterative, scores_median),
    ("Random Forest vs Median", scores_random_forest, scores_median),
]

for label, scores_a, scores_b in comparisons:
    result = corrected_paired_test(scores_a, scores_b, test_train_ratio)

    print(f"\n--- Corrected Paired t-Test: {label} ---")
    print(f"Mean Difference: {result['mean_diff']:.4f}")
    print(f"Wins: {result['wins']}")
    print(f"t-statistic: {result['t']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")

    save_test_result("results", f"GradientBoosting - {label}", result)