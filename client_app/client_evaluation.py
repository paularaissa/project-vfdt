# --- MÉTRICA KAPPA+ (TEMPORAL) ---
class KappaPlus:
    """
    Kappa+ (Temporal Kappa / Kappa Temp)
    Updates in a prequential manner.
    - y_pred is collected BEFORE training (test-then-train).
    - Baseline No-change: predicts the last observed true label.
    """
    def __init__(self):
        self.n = 0                      # Valid instances for calculation (starting from the 2nd instance).
        self.correct_model = 0          # correct model predictions
        self.correct_nochange = 0       # correct predictions of the No-change baseline
        self.prev_true = None           # last true y

    def update(self, y_true, y_pred_pretrain):
        # Updates the No-change baseline using the last true y
        if self.prev_true is not None:
            # The baseline predicts the previous label.
            y_pred_nc = self.prev_true
            self.n += 1
            if y_pred_pretrain == y_true:
                self.correct_model += 1
            if y_pred_nc == y_true:
                self.correct_nochange += 1
        # Stores the current label for the next iteration
        self.prev_true = y_true

    def get(self):
        if self.n == 0:
            return float("nan")
        acc = self.correct_model / self.n
        acc_nc = self.correct_nochange / self.n
        if acc_nc >= 1.0:
            # Avoids division by zero; if the baseline is perfect, there is no "margin for improvement."
            return 0.0
        return (acc - acc_nc) / (1.0 - acc_nc)


# --- KAPPA M ---
class KappaM:
    """
    KappaM
    Baseline predicts the most frequent class observed until the previous instance.
    Uses a prequential scheme: y_pred is the prediction BEFORE training on the instance.
    """
    def __init__(self):
        self.n = 0
        self.correct_model = 0
        self.correct_majority = 0
        self.counts = {}          # per-class counts: only updated AFTER comparing.
        self.prev_majority = None

    def update(self, y_true, y_pred_pretrain):
        y_pred_mc = self.prev_majority
        if y_pred_mc is not None:
            self.n += 1
            if y_pred_pretrain == y_true:
                self.correct_model += 1
            if y_pred_mc == y_true:
                self.correct_majority += 1
        self.counts[y_true] = self.counts.get(y_true, 0) + 1
        # Recalculates the majority class (ties: keeps the previous one if it is still a majority).
        if self.prev_majority is None or self.counts[y_true] >= self.counts.get(self.prev_majority, 0):
            # In case of a tie, switches to y_true (behavior is stable enough).
            self.prev_majority = y_true

    def get(self):
        if self.n == 0:
            return float("nan")
        acc = self.correct_model / self.n
        acc_maj = self.correct_majority / self.n
        if acc_maj >= 1.0:
            return 0.0
        return (acc - acc_maj) / (1.0 - acc_maj)


# --- BALANCED ACCURACY (on-line) ---
class BalancedAccuracyOnline:
    """
    Balanced Accuracy = mean recall by class.
    Updates the online confusion matrix with (y_true, y_pred_pretrain).
    """
    def __init__(self):
        self.tp = {}
        self.fn = {}
        self.labels_seen = set()

    def update(self, y_true, y_pred_pretrain):
        self.labels_seen.add(y_true)
        for lbl in [y_true, y_pred_pretrain]:
            if lbl is not None:
                self.tp.setdefault(lbl, 0)
                self.fn.setdefault(lbl, 0)
        if y_pred_pretrain == y_true:
            self.tp[y_true] = self.tp.get(y_true, 0) + 1
        else:
            self.fn[y_true] = self.fn.get(y_true, 0) + 1

    def get(self):
        if not self.labels_seen:
            return float("nan")
        recalls = []
        for c in self.labels_seen:
            tp = self.tp.get(c, 0)
            fn = self.fn.get(c, 0)
            den = tp + fn
            recalls.append(0.0 if den == 0 else tp / den)
        return sum(recalls) / len(recalls)


# --- G-MEAN (on-line) ---
class GMeanOnline:
    """
    G-Mean = (∏ recall_c)^(1/|C|)
    Sensitive to recall drops in rare classes.
    """
    def __init__(self):
        self.tp = {}
        self.fn = {}
        self.labels_seen = set()

    def update(self, y_true, y_pred_pretrain):
        self.labels_seen.add(y_true)
        for lbl in [y_true, y_pred_pretrain]:
            if lbl is not None:
                self.tp.setdefault(lbl, 0)
                self.fn.setdefault(lbl, 0)
        if y_pred_pretrain == y_true:
            self.tp[y_true] = self.tp.get(y_true, 0) + 1
        else:
            self.fn[y_true] = self.fn.get(y_true, 0) + 1

    def get(self):
        if not self.labels_seen:
            return float("nan")
        # if any recall=0 -> gmean=0
        prod = 1.0
        k = 0
        for c in self.labels_seen:
            tp = self.tp.get(c, 0)
            fn = self.fn.get(c, 0)
            den = tp + fn
            rc = 0.0 if den == 0 else tp / den
            prod *= rc
            k += 1
        return 0.0 if k == 0 else (prod ** (1.0 / k))
