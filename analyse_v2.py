# analyse_v2.py — version sans scipy/statsmodels
# Génère les graphiques individuels + paires (2 par 2) avec titres harmonisés au README

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ chemins / dossiers ------------------
CSV_FILE = "sleep_deprivation_dataset_detailed.csv"
FIG_DIR  = "figures"
OUT_DIR  = "results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

sns.set(style="whitegrid")

# ------------------ utilitaires ------------------
def find_col(df, candidates):
    """Retourne le premier vrai nom de colonne trouvé parmi 'candidates' (insensible casse/espaces/underscores)."""
    norm = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_")
        if key in norm:
            return norm[key]
    return None

def corr_np(x, y):
    """Corrélation de Pearson avec NumPy (sans scipy). Retourne r (float) ou np.nan si impossible."""
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 3:
        return np.nan
    r = np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1]
    return float(r)

def save_hist(series, title, fname, bins=20):
    plt.figure()
    sns.histplot(series.dropna(), bins=bins, kde=True)
    plt.title(title)  # = README
    plt.xlabel("Sleep Hours")
    plt.ylabel("Count")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"[OK] {title} -> {path}")

def save_regplot_single(df, x, y, title, fname, annotate_r=True, xlabel=None, ylabel=None):
    d = df[[x, y]].dropna()
    if d.empty:
        print(f"[SKIP] {title}: données vides")
        return
    r = corr_np(d[x], d[y])
    plt.figure()
    sns.regplot(x=x, y=y, data=d)
    plt.title(f"{title}\n r={r:.3f}" if annotate_r and not np.isnan(r) else title)  # = README
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"[OK] {title} -> {path}")

def save_regplot_pair(df, defs, pair_title, pair_fname):
    """
    defs = [
      (x_col, y_col, 'Titre du graphique (README = figure)', 'fichier_individuel.png', 'xlabel', 'ylabel'),
      (x_col, y_col, 'Titre du graphique (README = figure)', 'fichier_individuel.png', 'xlabel', 'ylabel')
    ]
    Crée aussi l'image combinée 2 colonnes avec titres propres.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax, (x, y, title, fname, xlabel, ylabel) in zip(axes, defs):
        d = df[[x, y]].dropna()
        if d.empty:
            ax.set_title(f"{title}\n(données manquantes)")
            ax.axis("off")
            continue
        r = corr_np(d[x], d[y])
        sns.regplot(x=x, y=y, data=d, ax=ax)
        ax.set_title(f"{title}\n r={r:.3f}" if not np.isnan(r) else title)  # = README
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)

        # Enregistre aussi la version individuelle harmonisée
        save_regplot_single(df, x, y, title, fname, annotate_r=True, xlabel=xlabel, ylabel=ylabel)

    fig.suptitle(pair_title, y=1.02, fontsize=14)
    path = os.path.join(FIG_DIR, pair_fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Pair -> {path}")

def numpy_ols(X, y):
    """OLS simple avec NumPy (ajoute une constante automatiquement). Retourne (beta, y_hat)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    Xc = np.column_stack([np.ones(len(X)), X])  # constante
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    y_hat = Xc @ beta
    return beta.flatten(), y_hat.flatten()

# ------------------ charger données ------------------
df = pd.read_csv(CSV_FILE)

# détecter les colonnes utiles
col_sleep   = find_col(df, ["Sleep_Hours"])
col_nback   = find_col(df, ["N_Back_Accuracy", "NBack_Accuracy"])
col_stroop  = find_col(df, ["Stroop_Task_Reaction_Time", "Stroop_Reaction_Time"])
col_pvt     = find_col(df, ["PVT_Reaction_Time"])
col_quality = find_col(df, ["Sleep_Quality_Score", "Sleep_Quality"])
col_caff    = find_col(df, ["Caffeine_Intake"])
col_stress  = find_col(df, ["Stress_Level"])
col_age     = find_col(df, ["Age"])
col_gender  = find_col(df, ["Gender"])

print("\n=== Colonnes détectées ===")
for k, v in {
    "Sleep_Hours": col_sleep, "N_Back_Accuracy": col_nback, "Stroop_RT": col_stroop,
    "PVT_RT": col_pvt, "Sleep_Quality": col_quality, "Caffeine_Intake": col_caff,
    "Stress_Level": col_stress, "Age": col_age, "Gender": col_gender,
}.items():
    print(f"{k:20s}: {v}")

# ------------------ Graphiques principaux (TITRES = README) ------------------
if col_sleep:
    save_hist(df[col_sleep], "Sleep Hours distribution", "distribution_sleeping_hours.png")

if col_sleep and col_nback:
    # Scatter principal
    save_regplot_single(
        df, col_sleep, col_nback,
        "Sleep Hours vs N-Back Accuracy (scatter)",
        "sleep_vs_performance.png",
        annotate_r=True,
        xlabel="Sleep Hours",
        ylabel="N-Back Accuracy"
    )
    # Tendance linéaire (même relation, titre harmonisé README)
    save_regplot_single(
        df, col_sleep, col_nback,
        "Linear Trend (regression)",
        "sleep_vs_n_back_accuracy_trend.png",
        annotate_r=True,
        xlabel="Sleep Hours",
        ylabel="N-Back Accuracy"
    )
    r_main = corr_np(df[col_sleep], df[col_nback])
    print(f"\nCorrélation Sleep_Hours ~ N_Back_Accuracy (r) = {r_main:.3f}")

# ------------------ Résultats additionnels (par paires, TITRES = README) ------------------
# 1) Reaction Time Tasks : Stroop / PVT (2x2 avec titres par figure)
pair_defs_rt = []
if col_sleep and col_stroop:
    pair_defs_rt.append((col_sleep, col_stroop,
                         "Sleep Hours vs Stroop Task Reaction Time",
                         "sleep_vs_stroop.png", "Sleep Hours", "Stroop Reaction Time"))
if col_sleep and col_pvt:
    pair_defs_rt.append((col_sleep, col_pvt,
                         "Sleep Hours vs PVT Reaction Time",
                         "sleep_vs_pvt.png", "Sleep Hours", "PVT Reaction Time"))

if len(pair_defs_rt) == 2:
    save_regplot_pair(df, pair_defs_rt, "Reaction Time Tasks", "pair_reaction_times.png")
else:
    # Enregistre individuellement si une seule dispo
    for (x, y, title, fname, xl, yl) in pair_defs_rt:
        save_regplot_single(df, x, y, title, fname, annotate_r=True, xlabel=xl, ylabel=yl)

# 2) N-Back check + by gender (barplot)
if col_sleep and col_nback:
    save_regplot_single(
        df, col_sleep, col_nback,
        "N-Back vs Sleep (Check)",
        "sleep_vs_nback_check.png",
        annotate_r=True,
        xlabel="Sleep Hours",
        ylabel="N-Back Accuracy"
    )

# barplot by gender
have_gender_plot = False
if col_gender and col_nback:
    d = df[[col_gender, col_nback]].dropna()
    if not d.empty:
        plt.figure()
        sns.barplot(x=col_gender, y=col_nback, data=d, ci="sd")
        plt.title("N-Back Accuracy by Gender")  # = README
        plt.xlabel("Gender"); plt.ylabel("N-Back Accuracy")
        plt.tight_layout()
        path = os.path.join(FIG_DIR, "nback_by_gender.png")
        plt.savefig(path, dpi=140)
        plt.close()
        print(f"[OK] N-Back Accuracy by Gender -> {path}")
        have_gender_plot = True

# Créer une paire combinée N-back check + gender (optionnel, pour aperçu 2x2)
if (col_sleep and col_nback) and have_gender_plot:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # gauche: regplot (N-back check)
    d = df[[col_sleep, col_nback]].dropna()
    r = corr_np(d[col_sleep], d[col_nback])
    sns.regplot(x=col_sleep, y=col_nback, data=d, ax=axes[0])
    axes[0].set_title(f"N-Back vs Sleep (Check)\n r={r:.3f}")
    axes[0].set_xlabel("Sleep Hours"); axes[0].set_ylabel("N-Back Accuracy")

    # droite: afficher l'image de barplot déjà générée
    img = plt.imread(os.path.join(FIG_DIR, "nback_by_gender.png"))
    axes[1].imshow(img); axes[1].axis("off")
    axes[1].set_title("N-Back Accuracy by Gender")

    fig.suptitle("N-Back Accuracy Checks", y=1.02, fontsize=14)
    path_pair = os.path.join(FIG_DIR, "pair_nback_gender.png")
    plt.savefig(path_pair, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Pair -> {path_pair}")

# 3) Age groups + effet partiel (NumPy OLS)
# a) Age groups
have_age_box = False
if col_age and col_nback:
    valid = df[[col_age, col_nback]].dropna().copy()
    if not valid.empty:
        valid["Age_Group"] = pd.qcut(valid[col_age], q=3, duplicates="drop")
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="Age_Group", y=col_nback, data=valid)
        plt.title("N-Back Accuracy by Age Group")  # = README
        plt.xlabel("Age Group"); plt.ylabel("N-Back Accuracy")
        plt.tight_layout()
        path = os.path.join(FIG_DIR, "nback_by_age_group.png")
        plt.savefig(path, dpi=140)
        plt.close()
        have_age_box = True
        print(f"[OK] N-Back Accuracy by Age Group -> {path}")

# b) Effet partiel via OLS NumPy (y ~ sleep + covariables)
have_partial = False
if col_nback and col_sleep:
    predictors = []
    names = []

    # Ajoute covariables si dispo
    for c, n in [
        (col_sleep, "Sleep_Hours"),
        (col_quality, "Sleep_Quality"),
        (col_caff, "Caffeine_Intake"),
        (col_stress, "Stress_Level"),
        (col_age, "Age"),
    ]:
        if c:
            predictors.append(df[c])
            names.append(n)

    X = pd.concat(predictors, axis=1) if predictors else None

    if X is not None and not X.empty:
        # Genre en dummies si dispo
        if col_gender and df[col_gender].notna().any():
            gd = pd.get_dummies(df[col_gender], prefix="Gender", drop_first=True)
            X = pd.concat([X, gd], axis=1)

        data = pd.concat([df[col_nback], X], axis=1).dropna()
        y = data[col_nback]
        X_use = data.drop(columns=[col_nback])

        if len(data) >= 10:
            # OLS NumPy
            X_arr = X_use.values
            y_arr = y.values
            beta, y_hat = numpy_ols(X_arr, y_arr)

            # Construire la droite « partielle » pour Sleep_Hours
            if "Sleep_Hours" in X_use.columns:
                ix = list(X_use.columns).index("Sleep_Hours")
                x_grid = np.linspace(X_use.iloc[:, ix].min(), X_use.iloc[:, ix].max(), 50)

                const = beta[0]
                contrib_others = 0.0
                for j, cname in enumerate(X_use.columns, start=1):
                    if cname == "Sleep_Hours":
                        continue
                    contrib_others += beta[j] * X_use[cname].mean()

                slope_sleep = beta[ix + 1]
                y_line = const + contrib_others + slope_sleep * x_grid

                # Scatter + ligne
                plt.figure(figsize=(6, 4))
                plt.scatter(X_use["Sleep_Hours"], y, s=18)
                plt.plot(x_grid, y_line, linewidth=2)
                plt.xlabel("Sleep Hours"); plt.ylabel("N-Back Accuracy")
                plt.title("Partial Sleep Effect (Regression)")  # = README
                plt.tight_layout()
                path = os.path.join(FIG_DIR, "nback_vs_sleep_partial_numpy.png")
                plt.savefig(path, dpi=140)
                plt.close()
                have_partial = True
                print(f"[OK] Partial Sleep Effect (Regression) -> {path}")

# Combiner Age group + Partial effect en paire (optionnel)
if have_age_box and have_partial:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    img1 = plt.imread(os.path.join(FIG_DIR, "nback_by_age_group.png"))
    axes[0].imshow(img1); axes[0].axis("off"); axes[0].set_title("N-Back Accuracy by Age Group")

    img2 = plt.imread(os.path.join(FIG_DIR, "nback_vs_sleep_partial_numpy.png"))
    axes[1].imshow(img2); axes[1].axis("off"); axes[1].set_title("Partial Sleep Effect (Regression)")

    fig.suptitle("Age and Regression Effects", y=1.02, fontsize=14)
    path_pair = os.path.join(FIG_DIR, "pair_age_partial.png")
    plt.savefig(path_pair, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Pair -> {path_pair}")

print("\nTerminé ✅  — Figures dans 'figures/' et résultats éventuels dans 'results/'.")
