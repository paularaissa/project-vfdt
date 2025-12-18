# --- MÉTRICA KAPPA+ (TEMPORAL) ---
class KappaPlus:
    """
    Kappa+ (Kappa Temporal / Kappa Temp)
    Atualiza de forma prequential.
    - y_pred do modelo é coletado ANTES do treino (test-then-train).
    - Baseline No-change: prevê o último rótulo verdadeiro visto.
    """
    def __init__(self):
        self.n = 0                      # instâncias válidas para o cálculo (a partir da 2ª)
        self.correct_model = 0          # acertos do modelo
        self.correct_nochange = 0       # acertos do baseline No-change
        self.prev_true = None           # último y verdadeiro

    def update(self, y_true, y_pred_pretrain):
        # Atualiza baseline No-change usando o último y_true
        if self.prev_true is not None:
            # baseline prevê o rótulo anterior
            y_pred_nc = self.prev_true
            self.n += 1
            if y_pred_pretrain == y_true:
                self.correct_model += 1
            if y_pred_nc == y_true:
                self.correct_nochange += 1
        # guarda rótulo atual para a próxima iteração
        self.prev_true = y_true

    def get(self):
        if self.n == 0:
            return float("nan")
        acc = self.correct_model / self.n
        acc_nc = self.correct_nochange / self.n
        if acc_nc >= 1.0:
            # evita divisão por zero; se baseline é perfeito, não há “margem de melhoria”
            return 0.0
        return (acc - acc_nc) / (1.0 - acc_nc)


# --- MÉTRICA KAPPA M (baseline Majority) ---
class KappaM:
    """
    KappaM (Kappa w.r.t. Majority).
    Baseline prevê a classe majoritária observada ATÉ a instância anterior.
    Usa esquema prequential: y_pred é a predição ANTES do treino na instância.
    """
    def __init__(self):
        self.n = 0
        self.correct_model = 0
        self.correct_majority = 0
        self.counts = {}          # contagem por classe, só atualiza DEPOIS de comparar
        self.prev_majority = None

    def update(self, y_true, y_pred_pretrain):
        # baseline majority com base nas contagens anteriores
        y_pred_mc = self.prev_majority
        if y_pred_mc is not None:
            self.n += 1
            if y_pred_pretrain == y_true:
                self.correct_model += 1
            if y_pred_mc == y_true:
                self.correct_majority += 1
        # atualiza contagem e majority APÓS usar como baseline
        self.counts[y_true] = self.counts.get(y_true, 0) + 1
        # recalcula majoritária (empate: mantém anterior se ainda for majoritária)
        if self.prev_majority is None or self.counts[y_true] >= self.counts.get(self.prev_majority, 0):
            # em empate, troca para y_true (comportamento estável o suficiente)
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
    Balanced Accuracy = mean recall por classe.
    Atualiza matriz de confusão on-line com (y_true, y_pred_pretrain).
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
    Sensível a queda de recall em classes raras.
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
        # se qualquer recall=0 -> gmean=0
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
