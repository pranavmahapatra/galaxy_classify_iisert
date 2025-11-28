import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from dataprep import data_r, data_c

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False
})

def latexify(name):
    greek = {
        "α": r"$\alpha$",
        "β": r"$\beta$",
        "γ": r"$\gamma$",
        "δ": r"$\delta$",
        "λ": r"$\lambda$",
        "σ": r"$\sigma$",
        "μ": r"$\mu$",
        "θ": r"$\theta$",
        "Θ": r"$\Theta$",
    }

    for g, rep in greek.items():
        name = name.replace(g, rep)

    return name

# data prep
data = data_r.iloc[:, 1:].copy()
data = data.drop(columns=["Mg", "log_10(Mdyn)", "log_10(Mbary)",
                          "log_10(Mgas)", "log_10(M*)"])


data = data.rename(columns=lambda c: latexify(c))

features = [col for col in data.columns if col != 'class']

# z-score each feature
for feature in features:
    mean = data[feature].mean()
    std  = data[feature].std()
    data[feature] = (data[feature] - mean) / std

palette = ["#007ea7", "#80ced7", "#9ad1d4", "#69d1c5"]
edge = "black"

# Violin plots per feature
for i, feature in enumerate(features):
    plt.figure(figsize=(6, 4))
    
    # Violin layer
    sns.violinplot(
        data=data,
        y="class",
        x=feature,
        hue="class",
        palette=palette,
        orient="h",
        linewidth=0,
        legend=False
    )

    # Box layer (with darker edges)
    sns.boxplot(
        data=data,
        y="class",
        x=feature,
        hue="class",
        palette=palette,
        orient="h",
        showcaps=False,
        width=0.1,
        boxprops={"facecolor": "black", "edgecolor": edge, "linewidth": 1.2},
        whiskerprops={"color": edge, "linewidth": 1.0},
        medianprops={"color": "red", "linewidth": 1.2},
        fliersize=0,
        legend=False
    )
    
    plt.axvline(0, color="gray", linestyle=(0, (8, 7)), linewidth=0.8)
    
    plt.title(feature)

    plt.tight_layout()
    plt.savefig(f"{i}.png", dpi=400, bbox_inches="tight")
    plt.close()

