import os
import warnings

import numpy as np
import scipy.stats as stats
from scipy.stats import wilcoxon
from scipy.linalg import hadamard
from joblib import Parallel, delayed
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import strlearn as sl
from river import drift
from river import naive_bayes

# ==========================================
# 1. KONFIGURATION
# ==========================================
CONFIG = {
    #'abrupt' or 'gradual'
    'drift_type': 'abrupt',

    # Extended benchmark
    'n_chunks': 250,
    'chunk_size': 200,
    'replications': 10,

    # Scenarios
    'scenario_features': [30, 60, 90],
    'scenario_drifts': [3, 5, 10, 15],

    # Parameter: gradual
    'gradual_spacing': 12,

    # Parameter: WHDD
    'padd_alpha': 0.07,
    'padd_theta': 0.19,
    'padd_r': 12,
    'padd_s': 50,
    'padd_e': 12,

    # Parameter: ADWIN
    'adwin_delta_error': 0.002,

    # Parameter: OCDD
    'ocdd_nu': 0.05,
    'ocdd_size': 300,
    'ocdd_percent': 0.3,
    'ocdd_kernel': 'rbf',
    'ocdd_gamma': 'scale',

    # Chart catalog
    'plots_dir': 'plots',
}

METHODS = [
    ("WHDD", "1"),
    ("ADWIN", "2"),
    ("EDDM", "3"),
    ("DDM", "4"),
    ("OCDD", "5"),
]


# ==========================================
# 2. WLASH DETECTOR
# ==========================================
class PADD_Walsh:
    def __init__(self, input_dim, alpha=0.07, theta=0.19, r=12, s=50, e=12, random_state=None):
        self.alpha = alpha
        self.theta = theta
        self.r = r
        self.s = s
        self.e = min(e, input_dim)
        self.rng = np.random.default_rng(random_state)

        next_p2 = 1 << (input_dim - 1).bit_length()
        H = hadamard(next_p2).astype(float) / np.sqrt(next_p2)

        self.W_direct = H[:input_dim, :self.e]
        self.b_direct = np.zeros(self.e, dtype=float)
        self.C = None

        df = 2 * self.s - 2
        self.t_crit = stats.t.ppf(1 - self.alpha / 2, df)

    def _relu(self, x):
        return np.maximum(0.0, x)

    def _forward_pass(self, X):
        X = np.asarray(X, dtype=float)
        return self._relu(np.dot(X, self.W_direct) + self.b_direct)

    def detect(self, X):
        c = self._forward_pass(X)
        drift_detected = False

        if self.C is not None:
            s_eff = min(self.s, c.shape[0], self.C.shape[0])

            if s_eff >= 2:
                idx_c = self.rng.integers(0, c.shape[0], size=(self.r, s_eff))
                idx_C = self.rng.integers(0, self.C.shape[0], size=(self.r, s_eff))

                cc = c[idx_c, :]
                pc = self.C[idx_C, :]

                mean_cc = np.mean(cc, axis=1)
                mean_pc = np.mean(pc, axis=1)
                var_cc = np.var(cc, axis=1, ddof=1)
                var_pc = np.var(pc, axis=1, ddof=1)

                var_pooled = (var_cc + var_pc) / 2.0
                t_stat = np.abs(mean_pc - mean_cc) / np.sqrt(var_pooled * (2.0 / s_eff) + 1e-12)

                a = np.sum(t_stat > self.t_crit)

                if a > self.theta * self.e * self.r:
                    drift_detected = True
                    self.C = None

        if self.C is None:
            self.C = c
        else:
            self.C = np.vstack((self.C, c))

        return drift_detected


# ==========================================
# 3. OCDD DETECTOR
# ==========================================
class OCDD:
    def __init__(self, nu=0.1, size=300, percent=0.3, kernel="rbf", gamma="scale"):
        self.nu = float(nu)
        self.size = int(size)
        self.percent = float(percent)
        self.kernel = kernel
        self.gamma = gamma

        self.model = None
        self.scaler = StandardScaler()  # We add a scaling object
        self.init_buffer = []
        self.window_data = []
        self.window_outlier = []
        self.drift_detected = False

    def _fit_model(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            self.model = None
            return

        # Skalujemy dane upewniając się, że średnia to 0, a wariancja to 1
        X_scaled = self.scaler.fit_transform(X)

        self.model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma
        )
        self.model.fit(X_scaled)

    def _predict_outlier_flag(self, x):
        if self.model is None:
            return 0

        # Przed predykcją nową próbkę również musimy przeskalować
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        pred = self.model.predict(x_scaled)[0]
        return 1 if pred == -1 else 0

    def update(self, x):
        x = np.asarray(x, dtype=float).ravel()
        self.drift_detected = False

        # Zbieranie początkowego bufora (również po twardym resecie)
        if self.model is None:
            self.init_buffer.append(x)
            if len(self.init_buffer) == self.size:
                self._fit_model(np.asarray(self.init_buffer))
                self.init_buffer = []
            return self

        if len(self.window_data) >= self.size:
            alpha = float(np.mean(self.window_outlier))

            if alpha >= self.percent:
                self.drift_detected = True

                # TWARDY RESET - ucinamy "pętlę śmierci"
                self.model = None
                self.window_data = []
                self.window_outlier = []
                self.init_buffer = [x]  # Zaczynamy zbierać od nowa
                return self
            else:
                self.window_data = self.window_data[1:]
                self.window_outlier = self.window_outlier[1:]

        outlier_flag = self._predict_outlier_flag(x)

        self.window_data.append(x)
        self.window_outlier.append(outlier_flag)

        return self
# ==========================================
# 4. DETECTOR CONSTRUCTORS
# ==========================================
def make_adwin(delta):
    return drift.ADWIN(delta=delta)


def make_eddm():
    if hasattr(drift, "binary") and hasattr(drift.binary, "EDDM"):
        return drift.binary.EDDM()
    if hasattr(drift, "EDDM"):
        return drift.EDDM()
    raise AttributeError("Nie znaleziono EDDM w river.drift.")


def make_ddm():
    if hasattr(drift, "binary") and hasattr(drift.binary, "DDM"):
        return drift.binary.DDM()
    if hasattr(drift, "DDM"):
        return drift.DDM()
    raise AttributeError("Nie znaleziono DDM w river.drift.")


def get_ocdd_percent(n_features):
    if n_features == 30:
        return 0.25
    if n_features == 60:
        return 0.30
    if n_features == 90:
        return 0.35
    raise ValueError(f"Brak zdefiniowanego ocdd_percent dla n_features={n_features}")


def make_ocdd(config, n_features):
    return OCDD(
        nu=config['ocdd_nu'],
        size=config['ocdd_size'],
        percent=get_ocdd_percent(n_features),
        kernel=config['ocdd_kernel'],
        gamma=config['ocdd_gamma'],
    )


# ==========================================
# 5. METRICS
# ==========================================
def calculate_D1(actual_drifts, detections):
    if not detections:
        return float('inf')
    distances = [min(abs(d - a) for a in actual_drifts) for d in detections]
    return float(np.mean(distances))


def calculate_D2(actual_drifts, detections):
    if not detections:
        return float('inf')
    distances = [min(abs(a - d) for d in detections) for a in actual_drifts]
    return float(np.mean(distances))


def calculate_R(n_actual, n_detections):
    if n_detections == 0:
        return float('inf')
    return float(abs((n_actual / n_detections) - 1.0))


# ==========================================
# 6. WILCOXON TEST
# ==========================================
def perform_wilcoxon(scores_A, scores_B):
    scores_A = np.asarray(scores_A, dtype=float)
    scores_B = np.asarray(scores_B, dtype=float)

    if np.all(scores_A == scores_B):
        return False

    if np.any(~np.isfinite(scores_A)) or np.any(~np.isfinite(scores_B)):
        mean_A = np.mean(scores_A)
        mean_B = np.mean(scores_B)
        return mean_A < mean_B

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, p_val = wilcoxon(scores_A, scores_B)
            return (p_val < 0.05) and (np.mean(scores_A) < np.mean(scores_B))
    except ValueError:
        return False


def get_significance_marker(metric_values_by_method, current_method_name, methods):
    current_scores = metric_values_by_method[current_method_name]
    wins = []

    for other_name, other_id in methods:
        if other_name == current_method_name:
            continue
        other_scores = metric_values_by_method[other_name]
        if perform_wilcoxon(current_scores, other_scores):
            wins.append(other_id)

    if len(wins) == 0:
        return "(-)"

    if len(wins) == len(methods) - 1:
        return "(all)"

    return "(" + ",".join(wins) + ")"


# ==========================================
# 7. AUXILIARY FUNCTIONS
# ==========================================
def build_generator_params(seed, n_features, n_drifts, n_chunks, chunk_size, drift_type, gradual_spacing):
    n_informative = max(1, int(n_features * 0.3))

    params = {
        'n_chunks': n_chunks,
        'chunk_size': chunk_size,
        'n_features': n_features,
        'n_informative': n_informative,
        'n_drifts': n_drifts,
        'random_state': seed
    }

    if drift_type == 'gradual':
        params['concept_sigmoid_spacing'] = gradual_spacing

    return params


def compute_drift_geometry(n_chunks, n_drifts, drift_type, gradual_spacing, window_p=0.10):
    """
    Geometria driftów zgodna z implementacją strlearn.StreamGenerator.

    Zwraca:
    - period_chunks: długość jednego okresu driftu w chunkach
    - actual_drifts: środki driftów w indeksach chunków
    - transition_windows: okna przejścia do wizualizacji
    - overlap_ratio: szerokość okna / period_chunks

    window_p = 0.10 oznacza, że dla gradual pokazujemy efektywne okno
    przejścia 10% -> 90% prawdopodobieństwa zmiany konceptu.
    """
    if n_drifts <= 0:
        return float(n_chunks), [], [], 0.0

    # W generatorze: period = n_samples / n_drifts.
    # Po przejściu na oś chunków daje to:
    period_chunks = n_chunks / float(n_drifts)

    # Środki driftów są w połowie kolejnych okresów:
    # 0.5*period, 1.5*period, 2.5*period, ...
    actual_drifts = [
        (i + 0.5) * period_chunks
        for i in range(n_drifts)
    ]

    if drift_type == 'gradual':
        css = float(gradual_spacing)
        if css <= 0:
            raise ValueError("gradual_spacing must be > 0")

        # Efektywna szerokość sigmoidy na poziomie p .. (1-p).
        # Dla p=0.10 dostajemy okno 10%-90%.
        z = abs(stats.logistic.ppf(window_p))
        width_chunks = (z / css) * period_chunks
        half_width = width_chunks / 2.0

        transition_windows = [
            (
                max(0.0, g - half_width),
                min(float(n_chunks - 1), g + half_width)
            )
            for g in actual_drifts
        ]
        overlap_ratio = width_chunks / period_chunks

    else:
        # abrupt = practically a point drift in the middle of the period
        transition_windows = [(g, g) for g in actual_drifts]
        overlap_ratio = 0.0

    return period_chunks, actual_drifts, transition_windows, overlap_ratio

def safe_mean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_std(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr))


def format_mean_std(values, width=18):
    mean_val = safe_mean(values)
    std_val = safe_std(values)

    if np.isinf(mean_val):
        text = "inf ± nan"
    elif np.isnan(mean_val):
        text = "nan ± nan"
    else:
        text = f"{mean_val:.3f} ± {std_val:.3f}"

    return f"{text:<{width}}"


# ==========================================
# 8. CHARTS
#    - single PNG: przebieg mean chunk error
#    - PNG: reprort (two panels)
# ==========================================
def plot_error_profile_only(repl_results, save_path):
    if not repl_results:
        return

    first = repl_results[0]
    n_chunks = first['config']['n_chunks']
    actual_drifts = first['geometry']['actual_drifts']
    transition_windows = first['geometry']['transition_windows']

    x = np.arange(n_chunks)

    error_matrix = np.asarray(
        [rep['series']['chunk_error_rates'] for rep in repl_results],
        dtype=float
    )
    mean_error = np.mean(error_matrix, axis=0)
    std_error = np.std(error_matrix, axis=0)

    fig, ax = plt.subplots(figsize=(15, 4.5))

    ax.plot(x, mean_error, label='Mean chunk error')
    ax.fill_between(x, mean_error - std_error, mean_error + std_error, alpha=0.2)

    for d in actual_drifts:
        ax.axvline(d, linestyle='--', linewidth=1)

    for start, end in transition_windows:
        if start != end:
            ax.axvspan(start, end, alpha=0.15)

    ax.set_xlabel('Chunk index', fontsize=18, fontweight='bold')
    ax.set_ylabel('Error rate', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.grid(True, alpha=0.3)
    ax.legend(prop={'size': 16, 'weight': 'bold'})

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_two_panel_report(repl_results, save_path):
    if not repl_results:
        return

    first = repl_results[0]
    n_chunks = first['config']['n_chunks']
    actual_drifts = first['geometry']['actual_drifts']
    transition_windows = first['geometry']['transition_windows']

    x = np.arange(n_chunks)

    error_matrix = np.asarray(
        [rep['series']['chunk_error_rates'] for rep in repl_results],
        dtype=float
    )
    mean_error = np.mean(error_matrix, axis=0)
    std_error = np.std(error_matrix, axis=0)

    detectors_to_plot = ['WHDD', 'ADWIN', 'EDDM', 'DDM', 'OCDD']
    detector_labels = {
        'WHDD': 'WHDD detections/count',
        'ADWIN':'ADWIN detections/count',
        'EDDM': 'EDDM detections/count',
        'DDM' : 'DDM detections/count',
        'OCDD': 'OCDD detections/count',
    }

    detection_counts = {}
    for det in detectors_to_plot:
        counts = np.zeros(n_chunks, dtype=int)
        for rep in repl_results:
            for d in rep['detections'][det]:
                if 0 <= d < n_chunks:
                    counts[d] += 1
        detection_counts[det] = counts

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    # ==========================================
    # TOP PANEL
    # ==========================================
    ax = axes[0]
    ax.plot(x, mean_error, label='Mean chunk error', linewidth=2.2)
    ax.fill_between(x, mean_error - std_error, mean_error + std_error, alpha=0.2)

    for d in actual_drifts:
        ax.axvline(d, linestyle='--', linewidth=1.5)

    for start, end in transition_windows:
        if start != end:
            ax.axvspan(start, end, alpha=0.15)

    ax.set_xlabel('Chunk index', fontsize=18, fontweight='bold')
    ax.set_ylabel('Error rate', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.grid(True, alpha=0.3)
    ax.legend(prop={'size': 16, 'weight': 'bold'})

    # ==========================================
    # BOTTOM PANEL
    # ==========================================
    ax = axes[1]

    for det in detectors_to_plot:
        counts = detection_counts[det]
        mask = counts > 0
        ax.scatter(
            x[mask],
            counts[mask],
            label=detector_labels[det],
            s=40
        )

    for d in actual_drifts:
        ax.axvline(d, linestyle='--', linewidth=1.5)

    for start, end in transition_windows:
        if start != end:
            ax.axvspan(start, end, alpha=0.15)

    max_count = max(np.max(v) for v in detection_counts.values()) if detection_counts else 1
    ax.set_ylim(-0.05, max_count + 0.25)

    ax.set_xlabel('Chunk index', fontsize=18, fontweight='bold')
    ax.set_ylabel('Detection count over replications', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.grid(True, alpha=0.3)
    ax.legend(prop={'size': 16, 'weight': 'bold'})

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ==========================================
# 9. REPLICATION
# ==========================================
def run_replication(seed, n_features, n_drifts, n_chunks, chunk_size, drift_type, gradual_spacing, config):
    generator_params = build_generator_params(
        seed=seed,
        n_features=n_features,
        n_drifts=n_drifts,
        n_chunks=n_chunks,
        chunk_size=chunk_size,
        drift_type=drift_type,
        gradual_spacing=gradual_spacing
    )

    stream = sl.streams.StreamGenerator(**generator_params)

    _, actual_drifts, transition_windows, overlap_ratio = compute_drift_geometry(
        n_chunks=n_chunks,
        n_drifts=n_drifts,
        drift_type=drift_type,
        gradual_spacing=gradual_spacing
    )

    detector_padd = PADD_Walsh(
        input_dim=n_features,
        alpha=config['padd_alpha'],
        theta=config['padd_theta'],
        r=config['padd_r'],
        s=config['padd_s'],
        e=config['padd_e'],
        random_state=seed
    )

    detector_adwin = make_adwin(config['adwin_delta_error'])
    detector_eddm = make_eddm()
    detector_ddm = make_ddm()
    detector_ocdd = make_ocdd(config, n_features)
    classifier_gnb = naive_bayes.GaussianNB()

    padd_detections = []
    adwin_detections = []
    eddm_detections = []
    ddm_detections = []
    ocdd_detections = []
    chunk_error_rates = []

    feature_keys = [f"f{j}" for j in range(n_features)]

    for chunk_idx in range(n_chunks):
        X_chunk, y_chunk = stream.get_chunk()

        if detector_padd.detect(X_chunk):
            padd_detections.append(chunk_idx)

        adwin_signaled_in_chunk = False
        eddm_signaled_in_chunk = False
        ddm_signaled_in_chunk = False
        ocdd_signaled_in_chunk = False
        errors_in_chunk = []

        for i in range(len(X_chunk)):
            x_arr = np.asarray(X_chunk[i], dtype=float)
            x_dict = dict(zip(feature_keys, x_arr))
            y_true = int(y_chunk[i])

            # Aktualizuj OCDD bez zatrzymywania
            detector_ocdd.update(x_arr)
            if detector_ocdd.drift_detected and not ocdd_signaled_in_chunk:
                ocdd_detections.append(chunk_idx)
                ocdd_signaled_in_chunk = True

            y_pred = classifier_gnb.predict_one(x_dict)
            if y_pred is None:
                y_pred = y_true

            error = 0 if y_pred == y_true else 1
            errors_in_chunk.append(error)

            classifier_gnb.learn_one(x_dict, y_true)

            detector_adwin.update(error)
            detector_eddm.update(error)
            detector_ddm.update(error)

            if detector_adwin.drift_detected and not adwin_signaled_in_chunk:
                adwin_detections.append(chunk_idx)
                adwin_signaled_in_chunk = True

            if detector_eddm.drift_detected and not eddm_signaled_in_chunk:
                eddm_detections.append(chunk_idx)
                eddm_signaled_in_chunk = True

            if detector_ddm.drift_detected and not ddm_signaled_in_chunk:
                ddm_detections.append(chunk_idx)
                ddm_signaled_in_chunk = True

        chunk_error_rates.append(float(np.mean(errors_in_chunk)))

    return {
        'config': {
            'seed': seed,
            'n_features': n_features,
            'n_drifts': n_drifts,
            'n_chunks': n_chunks,
            'chunk_size': chunk_size,
            'drift_type': drift_type,
            'gradual_spacing': gradual_spacing,
        },
        'geometry': {
            'actual_drifts': actual_drifts,
            'transition_windows': transition_windows,
            'overlap_ratio': overlap_ratio,
        },
        'series': {
            'chunk_error_rates': chunk_error_rates,
        },
        'detections': {
            'WHDD': padd_detections,
            'ADWIN': adwin_detections,
            'OCDD': ocdd_detections,
            'EDDM': eddm_detections,
            'DDM': ddm_detections,
        },
        'metrics': {
            'D1': {
                'WHDD': calculate_D1(actual_drifts, padd_detections),
                'ADWIN': calculate_D1(actual_drifts, adwin_detections),
                'EDDM': calculate_D1(actual_drifts, eddm_detections),
                'DDM': calculate_D1(actual_drifts, ddm_detections),
                'OCDD': calculate_D1(actual_drifts, ocdd_detections),
            },
            'D2': {
                'WHDD': calculate_D2(actual_drifts, padd_detections),
                'ADWIN': calculate_D2(actual_drifts, adwin_detections),
                'EDDM': calculate_D2(actual_drifts, eddm_detections),
                'DDM': calculate_D2(actual_drifts, ddm_detections),
                'OCDD': calculate_D2(actual_drifts, ocdd_detections),
            },
            'R': {
                'WHDD': calculate_R(n_drifts, len(padd_detections)),
                'ADWIN': calculate_R(n_drifts, len(adwin_detections)),
                'EDDM': calculate_R(n_drifts, len(eddm_detections)),
                'DDM': calculate_R(n_drifts, len(ddm_detections)),
                'OCDD': calculate_R(n_drifts, len(ocdd_detections)),
            }
        }
    }


# ==========================================
# 10. AGGREGATION
# ==========================================
def summarize_scenario(repl_results):
    methods = [m[0] for m in METHODS]
    metrics = ['D1', 'D2', 'R']

    summary = {metric: {} for metric in metrics}

    for metric in metrics:
        for method in methods:
            vals = [rep['metrics'][metric][method] for rep in repl_results]
            summary[metric][method] = vals

    return {
        'metrics': summary
    }


# ==========================================
# 11. PRINT TABLES
# ==========================================
def print_table_only(metric_name, all_results, drift_type):
    method_names = [m[0] for m in METHODS]
    method_labels = [f"{name} ({mid})" for name, mid in METHODS]

    print("=" * 205)
    print(f"{metric_name} | drift_type = {drift_type}")
    header = f"{'Scenariusz':<28} | "
    header += " | ".join(f"{label:<18}" for label in method_labels)
    print(header)
    print("-" * len(header))

    for scenario_name, result in all_results.items():
        metric_values = result['metrics'][metric_name]

        row_values = []
        for method_name in method_names:
            row_values.append(format_mean_std(metric_values[method_name], width=18))

        row1 = f"{scenario_name:<28} | " + " | ".join(row_values)
        print(row1)

        sig_values = []
        for method_name in method_names:
            marker = get_significance_marker(metric_values, method_name, METHODS)
            sig_values.append(f"{marker:<18}")

        row2 = f"{'':<28} | " + " | ".join(sig_values)
        print(row2)

    print("-" * len(header))


# ==========================================
# 12. SCENARIO CONSTRUCTION
# ==========================================
def build_scenarios(config):
    scenarios = []

    for n_drifts in config['scenario_drifts']:
        for n_features in config['scenario_features']:
            scenarios.append({
                'drift_type': config['drift_type'],
                'gradual_spacing': config['gradual_spacing'],
                'n_features': n_features,
                'n_drifts': n_drifts
            })

    return scenarios


# ==========================================
# 13. Main loop
# ==========================================
if __name__ == "__main__":
    n_chunks = CONFIG['n_chunks']
    chunk_size = CONFIG['chunk_size']
    replications = CONFIG['replications']

    scenarios = build_scenarios(CONFIG)
    all_results = {}

    for scenario in scenarios:
        drift_type = scenario['drift_type']
        gradual_spacing = scenario['gradual_spacing']
        n_features = scenario['n_features']
        n_drifts = scenario['n_drifts']

        scenario_name = f"{n_drifts} drifts | {n_features}F"

        repl_results = Parallel(n_jobs=-1)(
            delayed(run_replication)(
                42 + r,
                n_features,
                n_drifts,
                n_chunks,
                chunk_size,
                drift_type,
                gradual_spacing,
                CONFIG
            )
            for r in range(replications)
        )

        summary = summarize_scenario(repl_results)
        all_results[scenario_name] = summary

        # 1) single PNG panel - only error visualization
        error_plot_filename = f"{CONFIG['drift_type']}_{n_drifts}drifts_{n_features}F_error.png"
        error_plot_path = os.path.join(CONFIG['plots_dir'], error_plot_filename)
        plot_error_profile_only(repl_results, error_plot_path)

        # 2) PNG two panels
        report_plot_filename = f"{CONFIG['drift_type']}_{n_drifts}drifts_{n_features}F_report.png"
        report_plot_path = os.path.join(CONFIG['plots_dir'], report_plot_filename)
        plot_two_panel_report(repl_results, report_plot_path)

    print_table_only('D1', all_results, CONFIG['drift_type'])
    print_table_only('D2', all_results, CONFIG['drift_type'])
    print_table_only('R', all_results, CONFIG['drift_type'])