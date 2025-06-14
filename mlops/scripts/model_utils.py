from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def evaluate_model(y_true, y_pred, y_proba=None):
    """
    Returns evaluation metrics as a dictionary.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics