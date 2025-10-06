# analyse_v2.py (version sans SciPy ni statsmodels)
# Next Steps: cognitive measures (Stroop/PVT), multiple regression (NumPy), subgroup comparisons

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "sleep_deprivation_dataset_detailed.csv"
FIG_DIR = "figures"
OUT_DIR = "results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- utilitaires ----------
def find_col(df, candidates):
    """Retourne le premier nom de colonne existant (insensible casse/espaces/underscores)."""
    norm = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_")
        if key in norm:
            return norm[key]
    return None

def corr_np(x, y):
    """Corrélation de Pearson r sans SciPy (retourne seulement r)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return np.nan
    r = np.corrcoef(x, y)[0, 1]
    return float(r)

def save_regplot(df, x, y, title, fname, annotate_r=True):
    d = df[[x, y]].dropna()
    if d.empty:
        print(f"[SKIP] Plot {title}: données vides après dropna()")
        return
    r = corr_np(d[x], d[y])
    plt.figure()
    sns.regplot(x=x, y=y, data=d)
    subtitle = f"r={r:.3f}" if annotate_r and np.isfinite(r) else ""
    plt.title(f"{title}\n{subtitle}" if subtitle else title)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"[OK] Figure sauvegardée -> {path}")
    if np.isfinite(r):
        print(f"   Corrélation {x} ~ {y}: r={r:.3f}")

def ols_numpy(X: np.ndarray, y: np.ndarray, feat_names):
    """
    Régression linéaire OLS avec NumPy.
    Retourne: coefs, intercept, R2, y_pred, résumé texte
    """
    X1 = np.column_stack([np.ones(len(X)), X])        # ajoute l'intercept
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)     # solution OLS
    y_pred = X1 @ beta
    ss_tot = ((y - y.mean())**2).sum()
    ss_res = ((y - y_pred)**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    intercept = beta[0]
    coefs = beta[1:]
    lines = []
    lines.append("OLS (NumPy) — Multiple regression summary\n")
    lines.append(f"Samples: {len(y)}\n")
    lines.append("Predictors:\n")
    for n, c in zip(feat_names, coefs):
        lines.append(f"  - {n:25s} coef = {c: .4f}")
    lines.append(f"\nIntercept: {intercept: .4f}")
    lines.append(f"R^2: {r2:.4f}\n")
    return coefs, intercept, r2, y_pred, "\n".join(lines)

# ---------- charger les données ----------
df = pd.read_csv(CSV_FILE)

# détecter les colonnes utiles
col_sleep   = find_col(df, ["Sleep_Hours"])
col_nback   = find_col(df, ["N_Back_Accuracy", "NBack_Accuracy"])
col_stroop  = find_col(df, ["Stroop_Task_Reaction_Time", "Stroop_Reaction_Time"])
col_pvt     = find_col(df, ["PVT_Reaction_Time"])
col_quality = find_col(df, ["Sleep_Quality_Score", "Sleep_Quality"])
col_caffeine= find_col(df, ["Caffeine_Intake"])
col_stress  = find_col(df, ["Stress_Level"])
col_age     = find_col(df, ["Age"])
col_gender  = find_col(df, ["Gender"])

print("\n=== Colonnes détectées ===")
for label, c in [
    ("Sleep_Hours", col_sleep),
    ("N_Back_Accuracy", col_nback),
    ("Stroop_Task_Reaction_Time", col_stroop),
    ("PVT_Reaction_Time", col_pvt),
    ("Sleep_Quality_Score", col_quality),
    ("Caffeine_Intake", col_caffeine),
    ("Stress_Level", col_stress),
    ("Age", col_age),
    ("Gender", col_gender),
]:
    print(f"{label:28s}: {c}")

# ---------- 1) Autres mesures cognitives ----------
if col_sleep and col_stroop:
    save_regplot(df, col_sleep, col_stroop,
                 "Sleep Hours vs Stroop Task Reaction Time",
                 "sleep_vs_stroop.png", annotate_r=True)
else:
    print("[INFO] Stroop non tracé (colonnes manquantes)")

if col_sleep and col_pvt:
    save_regplot(df, col_sleep, col_pvt,
                 "Sleep Hours vs PVT Reaction Time",
                 "sleep_vs_pvt.png", annotate_r=True)
else:
    print("[INFO] PVT non tracé (colonnes manquantes)")

if col_sleep and col_nback:
    save_regplot(df, col_sleep, col_nback,
                 "Sleep Hours vs N-Back Accuracy (check)",
                 "sleep_vs_nback_check.png", annotate_r=True)

# ---------- 2) Régression multiple (NumPy) pour N_Back_Accuracy ----------
predictors = []
names = []
for c in [col_sleep, col_quality, col_caffeine, col_stress, col_age]:
    if c:
        predictors.append(df[c])
        names.append(c)

# encodage simple du genre si binaire (0/1)
if col_gender and df[col_gender].dropna().nunique() == 2:
    g_codes = pd.Categorical(df[col_gender]).codes.astype(float)
    g_codes[g_codes < 0] = np.nan  # -1 -> NaN
    predictors.append(pd.Series(g_codes, index=df.index, name=f"{col_gender}_bin"))
    names.append(f"{col_gender}_bin")

if col_nback and len(predictors) > 0:
    model_df = pd.concat([df[col_nback]] + predictors, axis=1).dropna()
    if len(model_df) < 10:
        print("[WARN] Trop peu d’observations pour OLS NumPy après dropna().")
    else:
        y = model_df[col_nback].to_numpy(dtype=float)
        X = model_df[names].to_numpy(dtype=float)
        coefs, intercept, r2, y_pred, summary = ols_numpy(X, y, names)
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, "ols_summary.txt")
        with open(out_path, "w") as f:
            f.write(summary)
        print(f"[OK] Résumé OLS (NumPy) sauvegardé -> {out_path}")

        # Effet partiel de Sleep_Hours
        if col_sleep in names:
            ix = names.index(col_sleep)
            X_mean = X.mean(axis=0)
            xs = np.linspace(X[:, ix].min(), X[:, ix].max(), 50)
            preds = []
            for xv in xs:
                v = X_mean.copy()
                v[ix] = xv
                preds.append(intercept + (v @ coefs))
            plt.figure()
            plt.scatter(X[:, ix], y, s=18)
            plt.plot(xs, preds, linewidth=2)
            plt.xlabel(col_sleep); plt.ylabel(col_nback)
            plt.title("N-Back vs Sleep (partial effect, NumPy OLS)")
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, "nback_vs_sleep_partial_numpy.png"), dpi=140)
            plt.close()
            print("[OK] Figure -> figures/nback_vs_sleep_partial_numpy.png")
else:
    print("[INFO] OLS NumPy non estimé (colonnes manquantes)")

# ---------- 3) Comparaisons par sous-groupes ----------
if col_gender and col_nback:
    grp = df[[col_gender, col_nback]].dropna().groupby(col_gender)[col_nback].agg(["mean", "std", "count"])
    print("\n=== N-Back par genre ===")
    print(grp)
    plt.figure()
    sns.barplot(x=col_gender, y=col_nback, data=df, ci="sd")
    plt.title("N-Back Accuracy by Gender (mean ± sd)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "nback_by_gender.png"), dpi=140)
    plt.close()
    print("[OK] Figure -> figures/nback_by_gender.png")

if col_age and col_nback:
    try:
        valid = df[[col_age, col_nback]].dropna()
        if not valid.empty:
            valid = valid.copy()
            valid["Age_Group"] = pd.qcut(valid[col_age], q=3, duplicates="drop")
            plt.figure()
            sns.boxplot(x="Age_Group", y=col_nback, data=valid)
            plt.title("N-Back Accuracy by Age Group")
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, "nback_by_age_group.png"), dpi=140)
            plt.close()
            print("[OK] Figure -> figures/nback_by_age_group.png")
    except Exception as e:
        print("[INFO] Age grouping skipped:", e)

print("\nTerminé ✅ — nouvelles figures dans 'figures/' et résumé OLS dans 'results/'.")
