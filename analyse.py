import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset
df = pd.read_csv("sleep_deprivation_dataset_detailed.csv")

# Vérifier les colonnes disponibles
print("Colonnes disponibles :", df.columns)
print("Colonnes :", list (df.columns))

# Étape 2 : trouver la bonne colonne cognitive automatiquement
candidates = [c for c in df.columns if 'cogn' in c.lower() or 'perform'in c.lower()]
print("Candidats cognition :", candidates)
ycol = "N_Back_Accuracy"
print ("colonne choisie pour Y (cognition) :", ycol)

# Aperçu des données
print("\nAperçu du dataset :")
print(df.head())

# Infos générales
print("\nInfos sur le dataset :")
print(df.info())

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(df.describe())

# Histogramme des heures de sommeil
sns.histplot(df['Sleep_Hours'], bins=20, kde=True)
plt.title("Distribution of sleeping hours")
plt.savefig("figures/distribution of sleeping hours.png")  # Sauvegarde dans ton dossier
plt.close()  # ferme la figure au lieu de l'afficher

# Scatter plot sommeil vs performance cognitive
sns.scatterplot (x='Sleep_Hours', y=ycol, data=df)
plt.title("Sleep vs performance")
plt.savefig("figures/Sleep vs performance.png")  # Sauvegarde
plt.close()

# Corrélation
corr = df['Sleep_Hours'].corr (df [ycol])
print("\nCorrélation sommeil ↔ performance cognitive :", corr)

# Regression line plot (Sleep vs Performance)
g = sns.lmplot(x="Sleep_Hours", y=ycol, data=df, ci=95)
g.fig.suptitle(f"Linear Trend: Sleep Hours vs {ycol}", y=0.95)
g.fig.subplots_adjust(top=0.9)  # <-- ajoute un peu d’espace
g.set_xlabels("Sleep Hours")
g.set_ylabels(ycol)
g.fig.tight_layout()
g.fig.savefig(f"figures/sleep_vs_{ycol}_trend.png")
plt.close()

print(f"\nCorrelation Sleep_Hours vs {ycol} = {corr:.3f}")
if abs(corr) < 0.2:
    print("Interpretation: Very weak correlation → no clear linear relationship.")
elif corr > 0:
    print("Interpretation: Positive correlation → more sleep is linked with higher performance.")
else:
    print("Interpretation: Negative correlation → more sleep is linked with lower performance.")