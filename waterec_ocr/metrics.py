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
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Need equally-sized nonempty labels and predictions")
    correct = sum(price_value(gt) is not None and price_value(gt) == price_value(pred)
                  for gt, pred in zip(labels, predictions))
    return {"numeric_correct": correct, "numeric_accuracy": correct / len(labels),
            "numeric_reference_parseable": sum(price_value(gt) is not None for gt in labels),
            "numeric_metric": "Whole string decimal with optional DH; unparseable values count incorrect"}


def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def score_pairs(labels, predictions, normalization="upper_no_whitespace"):
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Need equally-sized nonempty labels and predictions")
    correct = edits = characters = alphanumeric = 0
    for gt, pred in zip(labels, predictions):
        gt, pred = normalize(gt, normalization), normalize(pred, normalization)
        correct += gt == pred
        edits += edit_distance(gt, pred)
        characters += len(gt)
        clean = lambda text: "".join(c for c in text if c.isalnum())
        alphanumeric += clean(gt) == clean(pred)
    return {"n": len(labels), "correct": correct, "exact_accuracy": correct/len(labels),
            "cer": edits/max(characters, 1), "alphanumeric_exact_accuracy": alphanumeric/len(labels)}
