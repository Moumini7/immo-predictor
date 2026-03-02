import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sys, os
from models.model import  Styles_manager

# Charger le CSS
Styles_manager.load_css()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import DataLoader, get_correlation_with_price, REG_FEATURES, TARGET_REG, TARGET_CLF

# Titre du dashboard
st.markdown("""
<div class="metric-card metric-card-green">
    <h1 style="margin-bottom: -15px;">
        EDA
    </h1> 
    <span style="margin-right: 10px;"> 
        Analyse exploratoire de données 
    </span>
</div>
""", unsafe_allow_html=True)
st.divider()


@st.cache_data
def load():
    return DataLoader.load("data/train.csv")

df = load()


st.subheader("1. Résumé global du dataset")
summary = DataLoader.summary(df)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Maisons",       summary["n_rows"])
c2.metric("Variables",     summary["n_cols"])
c3.metric("Valeurs manq.", summary["missing_total"])
c4.metric("Prix moyen",   f"{summary['price_mean']:,.0f} $")
c5.metric("Prix médian",  f"{summary['price_median']:,.0f} $")

st.divider()


st.subheader("2 Aperçu des données")
n = st.slider("Nombre de lignes à afficher", 5, 50, 10)
st.dataframe(df.head(n), use_container_width=True)

with st.expander("Types de colonnes"):
    col_info = pd.DataFrame({
        "Type": df.dtypes,
        "Valeurs uniques": df.nunique(),
        "Manquants": df.isnull().sum(),
        "Manquants (%)": (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(col_info, use_container_width=True)

st.divider()


st.subheader("3 Distribution de SalePrice (variable cible)")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[TARGET_REG], bins=60, color="#1f8828", edgecolor="white", alpha=0.85)
    ax.set_title("Distribution de SalePrice", fontsize=13, fontweight='bold')
    ax.set_xlabel("Prix de vente ($)")
    ax.set_ylabel("Fréquence")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    st.pyplot(fig)
    st.caption("Distribution asymétrique à droite (right-skewed).")

with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log1p(df[TARGET_REG]), bins=60, color="#7ee039", edgecolor="white", alpha=0.85)
    ax.set_title("Distribution de log(SalePrice)", fontsize=13, fontweight='bold')
    ax.set_xlabel("log(Prix de vente)")
    ax.set_ylabel("Fréquence")
    st.pyplot(fig)
    st.caption("Après transformation log → distribution plus normale.")

st.divider()

st.subheader("4 Corrélation des variables numériques avec SalePrice")

corr = get_correlation_with_price(df)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#246d15" if x > 0 else "#82c02b" for x in corr.values]
corr.plot(kind='barh', ax=ax, color=colors)
ax.set_title("Corrélation de Pearson avec SalePrice", fontsize=13, fontweight='bold')
ax.set_xlabel("Corrélation")
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
st.pyplot(fig)
st.caption("Variables en bleu = corrélation positive. En rouge = corrélation négative.")

st.divider()

st.subheader("5️ Relations entre variables clés et SalePrice")

top4 = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF']
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()
for i, feat in enumerate(top4):
    axes[i].scatter(df[feat], df[TARGET_REG], alpha=0.3, color="#246d15", s=10)
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel("SalePrice ($)")
    axes[i].set_title(f"{feat} vs SalePrice")
    axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
plt.tight_layout()
st.pyplot(fig)

st.divider()

st.subheader("6 Valeurs manquantes")

miss_report = DataLoader.missing_report(df)
if miss_report.empty:
    st.success("Aucune valeur manquante !")
else:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(miss_report, use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(8, max(4, len(miss_report) * 0.35)))
        miss_report["Pourcentage (%)"].sort_values().plot(
            kind='barh', ax=ax, color="#246d15", alpha=0.8
        )
        ax.set_title("% de valeurs manquantes par colonne")
        ax.set_xlabel("Pourcentage (%)")
        plt.tight_layout()
        st.pyplot(fig)

st.divider()

st.subheader("7 Distribution de BldgType (cible Classification)")

bldg_counts = df[TARGET_CLF].value_counts()

col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(
        pd.DataFrame({"Type": bldg_counts.index, "Nombre": bldg_counts.values,
                      "%": (bldg_counts.values / len(df) * 100).round(2)}),
        use_container_width=True, hide_index=True
    )
with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    bldg_counts.plot(kind='bar', ax=ax, color="#246d15", edgecolor='white', alpha=0.85)
    ax.set_title("Distribution de BldgType", fontsize=13, fontweight='bold')
    ax.set_xlabel("Type de bâtiment")
    ax.set_ylabel("Nombre de maisons")
    ax.set_xticklabels(bldg_counts.index, rotation=30)
    plt.tight_layout()
    st.pyplot(fig)

st.caption("**1Fam** (maison individuelle) est très majoritaire → dataset déséquilibré pour la classification.")

st.divider()

# ── 8. Statistiques descriptives ─────────────────────────────────────────────
st.subheader("8 Statistiques descriptives")
st.dataframe(df[REG_FEATURES + [TARGET_REG]].describe().T.round(2), use_container_width=True)
