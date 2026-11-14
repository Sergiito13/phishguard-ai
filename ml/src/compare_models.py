"""
PhishGuard AI - Model Comparison

Compare:
- Logistic Regression
- HistGradientBoosting

Evaluate using the validation dataset only.

The test dataset remains untouched.
"""

from pathlib import Path

import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from train_model import (
    load_dataset,
    create_model,
    TRAIN_FILE,
    VALIDATION_FILE,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

SOURCES = [
    "legitphish",
    "phiusiil",
    "phishbd",
    "url_phish",
]


# ============================================================
# 2. CREATE MODELS
# ============================================================

def create_models():

    return {
        "Logistic Regression": create_model(),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            learning_rate=0.08,
            l2_regularization=5.0,
            random_state=RANDOM_STATE,
        ),
    }


# ============================================================
# 3. EVALUATE PREDICTIONS
# ============================================================

def calculate_metrics(y_true, y_pred, probabilities):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy_score(
            y_true, y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        "average_precision": average_precision_score(
            y_true,
            probabilities,
        ),

        "false_positives": int(fp),

        "false_negatives": int(fn),
    }


# ============================================================
# 4. PRINT METRICS
# ============================================================

def print_metrics(metrics):

    print(
        f"Accuracy:          {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:         {metrics['precision']:.4f}"
    )

    print(
        f"Recall:            {metrics['recall']:.4f}"
    )

    print(
        f"F1-score:          {metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC:           {metrics['roc_auc']:.4f}"
    )

    print(
        f"Average Precision: {metrics['average_precision']:.4f}"
    )

    print(
        f"False positives:   {metrics['false_positives']:,}"
    )

    print(
        f"False negatives:   {metrics['false_negatives']:,}"
    )


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PHISHGUARD AI - MODEL COMPARISON")
    print("=" * 65)

    # Load datasets

    X_train, y_train = load_dataset(
        TRAIN_FILE
    )

    X_val, y_val = load_dataset(
        VALIDATION_FILE
    )

    metadata = pd.read_csv(
        VALIDATION_FILE,
        usecols=["sources"],
        keep_default_na=False,
    )

    if len(metadata) != len(y_val):
        raise ValueError(
            "Validation metadata is misaligned"
        )

    source_lists = (
        metadata["sources"]
        .str.split("|", regex=False)
    )

    source_masks = {
        source: source_lists.apply(
            lambda values: source in values
        ).to_numpy()
        for source in SOURCES
    }

    print(
        f"\nTraining URLs: {len(X_train):,}"
    )

    print(
        f"Validation URLs: {len(X_val):,}"
    )

    models = create_models()

    results = []

    # --------------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------------

    for model_name, model in models.items():

        print("\n" + "=" * 65)
        print(f"MODEL: {model_name}")
        print("=" * 65)

        print("\nTraining...")

        model.fit(
            X_train,
            y_train,
        )

        print("Training completed.")

        predictions = model.predict(
            X_val
        )

        probabilities = model.predict_proba(
            X_val
        )[:, 1]

        # ----------------------------------------------------
        # OVERALL EVALUATION
        # ----------------------------------------------------

        print("\nOVERALL VALIDATION RESULTS")

        overall_metrics = calculate_metrics(
            y_val,
            predictions,
            probabilities,
        )

        print_metrics(overall_metrics)

        results.append({
            "model": model_name,
            "source": "overall",
            **overall_metrics,
        })

        # ----------------------------------------------------
        # DATASET SOURCE EVALUATION
        # ----------------------------------------------------

        for source in SOURCES:

            mask = source_masks[source]

            y_source = y_val[mask]

            pred_source = predictions[mask]

            prob_source = probabilities[mask]

            if len(y_source) == 0:
                continue

            print(
                f"\nSOURCE: {source.upper()}"
            )

            metrics = calculate_metrics(
                y_source,
                pred_source,
                prob_source,
            )

            print_metrics(metrics)

            results.append({
                "model": model_name,
                "source": source,
                **metrics,
            })

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)

    columns = [
        "model",
        "source",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "false_negatives",
    ]

    print(
        results_df[columns].to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
            },
        )
    )

    print("\nComparison completed.")
    print("Test dataset has not been used.")


# ============================================================
# 6. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()