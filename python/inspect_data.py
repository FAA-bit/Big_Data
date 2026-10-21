
#!/usr/bin/env python3
"""Inspect, analyze, visualize, and model the AI4I 2020 dataset."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = CODE_DIR / "data" / "ai4i2020.csv"


# --------------------------------------------------
# 2. Load and inspect the dataset
# --------------------------------------------------

def main() -> None:
    """Run the complete AI4I analysis."""

    print("=" * 60)
    print("AI4I 2020 - DATA ANALYSIS")
    print("=" * 60)

    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found: {DATA_FILE}")
        print("Check that ai4i2020.csv is inside the data folder.")
        return

    print("\nStep 1: Loading dataset...")
    data = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

    print("Dataset loaded successfully.")
    print("File:", DATA_FILE)

    print("\nStep 2: First 10 rows")
    print(data.head(10).to_string(index=False))

    print("\nStep 3: Dataset size")
    print("Rows:", data.shape[0])
    print("Columns:", data.shape[1])

    print("\nStep 4: Data types")
    print(data.dtypes)

    print("\nStep 5: Missing values")
    print(data.isna().sum())

    print("\nStep 6: Duplicate rows")
    print("Duplicates:", data.duplicated().sum())

    # Check that the columns needed for analysis exist.
    required_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
        "Type",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        print("ERROR: Required columns are missing:", missing_columns)
        return

    if data[required_columns].isna().any().any():
        print("ERROR: Required columns contain missing values.")
        print("Handle missing values before training the model.")
        return

    # --------------------------------------------------
    # 3. Machine failure analysis
    # --------------------------------------------------

    print("\nStep 7: Machine failures")

    failure_count = int(data["Machine failure"].sum())
    failure_rate = data["Machine failure"].mean()

    print("Recorded failures:", failure_count)
    print("Failure rate:", f"{failure_rate:.2%}")

    print("\nStep 8: Tool wear comparison")

    failure_data = data[data["Machine failure"] == 1]
    normal_data = data[data["Machine failure"] == 0]

    print("\nTool wear with recorded failures:")
    print(failure_data["Tool wear [min]"].describe())

    print("\nTool wear without recorded failures:")
    print(normal_data["Tool wear [min]"].describe())

    # --------------------------------------------------
    # 4. Feature engineering: temperature difference
    # --------------------------------------------------

    print("\nStep 9: Calculate temperature difference")

    data["Temperature difference [K]"] = (
        data["Process temperature [K]"]
        - data["Air temperature [K]"]
    )

    print(
        data[
            [
                "Air temperature [K]",
                "Process temperature [K]",
                "Temperature difference [K]",
            ]
        ].head(10).to_string(index=False)
    )

    print(
        "\nAverage temperature difference:",
        f"{data['Temperature difference [K]'].mean():.2f} K",
    )

    print(
        "Maximum temperature difference:",
        f"{data['Temperature difference [K]'].max():.2f} K",
    )

    # --------------------------------------------------
    # 5. Grouping and aggregation
    # --------------------------------------------------

    print("\nStep 10: Average torque by machine type")

    torque_by_type = data.groupby("Type")["Torque [Nm]"].mean()
    print(torque_by_type)

    # --------------------------------------------------
    # 6. Visualize temperature differences
    # --------------------------------------------------

    print("\nStep 11: Creating temperature histogram...")

    plt.figure(figsize=(9, 5))
    plt.hist(data["Temperature difference [K]"], bins=20)
    plt.title("Distribution of Temperature Differences")
    plt.xlabel("Temperature Difference [K]")
    plt.ylabel("Number of Observations")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 7. Train and evaluate the ML model
    # --------------------------------------------------

    run_random_forest(data)


def run_random_forest(data: pd.DataFrame) -> None:
    """Train a Random Forest and compare it with a baseline."""

    print("\n" + "=" * 60)
    print("MACHINE LEARNING: RANDOM FOREST")
    print("=" * 60)

    # Use operating measurements as features.
    # Do not include identifiers or failure-type columns.
    features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    X = data[features]
    y = data["Machine failure"]

    # Split data into training and test sets.
    print("\nStep 12: Splitting training and test data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training observations:", len(X_train))
    print("Test observations:", len(X_test))
    print("Training Random Forest...")

    # Train the model.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Training completed!")

    # Predict the test observations.
    print("\nStep 13: Evaluating Random Forest...")

    y_pred = model.predict(X_test)

    # Create the baseline: always predict no failure.
    baseline_pred = [0] * len(y_test)

    # Calculate evaluation metrics.
    def calculate_metrics(y_true, predictions):
        return {
            "Accuracy": accuracy_score(y_true, predictions),
            "Precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_true, predictions, zero_division=0
            ),
        }

    model_metrics = calculate_metrics(y_test, y_pred)
    baseline_metrics = calculate_metrics(y_test, baseline_pred)

    # Display a side-by-side comparison.
    print("\nMODEL COMPARISON")
    print("-" * 56)
    print(f"{'Metric':<15}{'Random Forest':>20}{'Baseline':>18}")
    print("-" * 56)

    for metric_name in ["Accuracy", "Precision", "Recall"]:
        model_value = model_metrics[metric_name]
        baseline_value = baseline_metrics[metric_name]

        print(
            f"{metric_name:<15}"
            f"{model_value:>19.2%}"
            f"{baseline_value:>17.2%}"
        )

    # Show how many failures were detected or missed.
    print("\nStep 14: Confusion matrix")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    print("True negatives (correct normal):", tn)
    print("False positives (false alarms):", fp)
    print("False negatives (missed failures):", fn)
    print("True positives (detected failures):", tp)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No failure", "Failure"],
    )

    display.plot()
    plt.title("Random Forest - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
