"""Exact OCR scoring; country-format correction is deliberately absent."""
from .data_io import normalize
from decimal import Decimal
import re


def price_value(text):
    """Strict benchmark value: decimal point/comma and optional DH suffix.

    No substring extraction or removal of decimal punctuation. Other currency
    formats remain unparsed and should use a separately documented metric.
    """
    text = "".join(text.upper().split())
    if not re.fullmatch(r"\d+(?:[.,]\d{1,2})?(?:DH)?", text):
        return None
    return Decimal(text.removesuffix("DH").replace(",", "."))


def score_price_values(labels, predictions):
    raise ValueError("Numeric-equivalence reporting is disabled; use literal score_pairs")


def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


EXACT_POLICY = "literal_string_exact_v1"


def require_exact_selection(state, where="training state"):
    if state.get("selection_policy") != EXACT_POLICY:
        raise ValueError(f"{where} uses a legacy/incompatible selection policy; choose a new output directory. Old normalized/numeric/edit-distance scores cannot be resumed as literal exact.")
    return state


def selection_score(metrics):
    if metrics.get("metric_policy") != EXACT_POLICY:
        raise ValueError("Checkpoint selection requires literal-string exact metrics")
    return (metrics["correct"], 0)  # Earliest epoch wins an exact-count tie.


def score_pairs(labels, predictions, normalization="none"):
    """Literal equality only: no case, whitespace, Unicode or numeric conversion."""
    if normalization not in (None, "none"):
        raise ValueError("Scoring is literal exact only; normalization must be none")
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Need equally-sized nonempty labels and predictions")
    if any(not isinstance(x, str) for x in list(labels) + list(predictions)):
        raise TypeError("Labels and predictions must be strings")
    correct = sum(gt == pred for gt, pred in zip(labels, predictions))
    return {"n": len(labels), "correct": correct, "exact_accuracy": correct / len(labels),
            "metric_policy": EXACT_POLICY}
