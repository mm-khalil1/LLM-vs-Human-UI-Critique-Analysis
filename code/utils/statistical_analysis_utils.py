import warnings

import numpy as np
import pandas as pd
from pingouin import intraclass_corr
from scipy.stats import kstest, shapiro, ttest_rel
from sklearn.metrics import confusion_matrix
from statsmodels.multivariate.manova import MANOVA


warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

__all__ = [
    "bp_kappa",
    "bp_kappa_manually_defined",
    "compute_icc",
    "compute_manova",
    "test_normality",
    "ttest_rel",
    "_weights_matrix",
]


def test_normality(data) -> None:
    """Run Shapiro-Wilk and Kolmogorov-Smirnov normality checks."""
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise ValueError("Input data must be an array-like object.")

    _, p_value_sw = shapiro(data)
    _, p_value_ks = kstest(data, "norm")
    p_value_sw = round(p_value_sw, 4)
    p_value_ks = round(p_value_ks, 4)

    alpha = 0.05

    if p_value_sw > alpha:
        print(p_value_sw, "Shapiro-Wilk Test: Data looks normally distributed (fail to reject H0)")
    else:
        print(p_value_sw, "Shapiro-Wilk Test: Data does NOT look normally distributed (reject H0)")

    if p_value_ks > alpha:
        print(p_value_ks, "Kolmogorov-Smirnov Test: Data looks normally distributed (fail to reject H0)")
    else:
        print(p_value_ks, "Kolmogorov-Smirnov Test: Data does NOT look normally distributed (reject H0)")


def _weights_matrix(labels, weights_type="ordinal") -> np.ndarray:
    """Build a weight matrix for ordinal agreement statistics."""
    labels = list(labels)
    q = len(labels)
    if q == 0:
        raise ValueError("Length of labels is 0.")

    max_label = max(labels)
    min_label = min(labels)
    if max_label == min_label:
        return np.ones((q, q))

    weights = np.zeros((q, q))

    for i in range(q):
        for j in range(q):
            if weights_type == "linear":
                weights[i, j] = 1 - abs(labels[i] - labels[j]) / (max_label - min_label)
            elif weights_type == "quadratic":
                weights[i, j] = 1 - (abs(labels[i] - labels[j]) / (max_label - min_label)) ** 2
            elif weights_type == "ordinal":
                nij = max(labels[i], labels[j]) - min(labels[i], labels[j]) + 1
                weights[i, j] = nij * (nij - 1) / 2
            else:
                raise ValueError("Invalid weights_type. Choose 'linear', 'quadratic', or 'ordinal'.")

    if weights_type == "ordinal":
        weights = 1 - weights / np.max(weights)

    return weights


def bp_kappa(rater1, rater2, labels, weights_type="ordinal"):
    """Calculate Brennan-Prediger kappa between two raters."""
    if len(rater1) != len(rater2):
        raise ValueError("Rater lists must be of equal length.")

    cm = confusion_matrix(rater1, rater2, labels=labels)
    w = _weights_matrix(labels, weights_type)
    n = len(rater1)
    q = len(labels)

    pa = np.sum(w * cm) / n
    pe = w.sum() / q**2
    return (pa - pe) / (1 - pe)


bp_kappa_manually_defined = bp_kappa


def compute_icc(df, common_idx) -> pd.DataFrame:
    """Compute ICC values after converting paired rater columns to long format."""
    df_long = df.melt(id_vars=[common_idx], var_name="Rater", value_name="Score")
    return intraclass_corr(
        data=df_long,
        targets=common_idx,
        raters="Rater",
        ratings="Score",
    )


def compute_manova(df, metrics):
    """Run MANOVA comparing human and model ratings across metrics."""
    df_human = df[metrics].copy()
    df_human["source"] = "Human"

    df_model = df[[f"{metric}_llm" for metric in metrics]].copy()
    df_model.columns = metrics
    df_model["source"] = "Model"

    df_all = pd.concat([df_human, df_model], ignore_index=True)
    manova = MANOVA.from_formula(" + ".join(metrics) + " ~ source", data=df_all)
    return manova.mv_test()
