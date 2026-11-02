
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate_cv(model, X, y, groups, n_splits=5, verbose=True):
    """5-fold cross-validation split by engine.
 
    Returns (mean RMSE, out-of-fold predictions). Out-of-fold = each row is
    predicted by the model that did NOT see its engine during training.
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y))
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        m = clone(model)                 
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = m.predict(X.iloc[va_idx])
        scores.append(rmse(y.iloc[va_idx], oof[va_idx]))
        if verbose:
            print(f"  fold {fold + 1}: RMSE = {scores[-1]:.2f}")
    if verbose:
        print(f"  mean RMSE = {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    return float(np.mean(scores)), oof

def nasa_score(y_true, y_pred):
    """NASA asymmetric score: late predictions (d > 0) cost more. Lower is better."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))