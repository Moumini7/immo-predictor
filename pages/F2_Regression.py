import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import DataLoader, build_and_train_all, REG_FEATURES, TARGET_REG
from models.model import  Styles_manager

# Charger le CSS
Styles_manager.load_css()


# Titre du dashboard
st.markdown("""
<div class="metric-card metric-card-green">
    <h1 style="margin-bottom: -15px;">
        REGRESSION
    </h1> 
    <span style="margin-right: 10px;"> 
        Régression – Prédiction du Prix de Vente 
    </span>
</div>
""", unsafe_allow_html=True)
st.divider()

@st.cache_data
def load():
    return DataLoader.load("data/train.csv")

@st.cache_resource
def train():
    df = DataLoader.load("data/train.csv")
    return build_and_train_all(df)

df = load()

with st.spinner("Entraînement des modèles en cours..."):
    prep, reg_model, clf_model, reg_data, clf_data = train()

st.success("Modèles entraînés !")
st.divider()


with st.expander("Variables utilisées pour la régression"):
    st.markdown(", ".join([f"`{f}`" for f in REG_FEATURES]))

st.divider()

st.subheader("1 Métriques d'évaluation")

results = reg_data["results"]
y_test  = reg_data["y_test"]

# Tableau comparatif
rows = []
for name, m in results.items():
    rows.append({"Modèle": name, "MAE ($)": f"{m['MAE']:,.0f}", "RMSE ($)": f"{m['RMSE']:,.0f}", "R²": f"{m['R2']:.4f}"})

df_res = pd.DataFrame(rows)
st.dataframe(df_res, use_container_width=True, hide_index=True)

# Métriques visuelles
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("DT – MAE",  f"{results['Decision Tree']['MAE']:,.0f} $")
col2.metric("DT – RMSE", f"{results['Decision Tree']['RMSE']:,.0f} $")
col3.metric("DT – R²",   f"{results['Decision Tree']['R2']:.4f}")
col4.metric("RF – MAE",  f"{results['Random Forest']['MAE']:,.0f} $")
col5.metric("RF – RMSE", f"{results['Random Forest']['RMSE']:,.0f} $")
col6.metric("RF – R²",   f"{results['Random Forest']['R2']:.4f}")

st.caption("Le **Random Forest** donne en général de meilleures performances grâce à l'agrégation de plusieurs arbres.")

st.divider()


st.subheader("2 Visualisations")

tab1, tab2, tab3 = st.tabs(["Prédit vs Réel", "Résidus", "[Importance des varialbes]")

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, (name, color) in enumerate([("Decision Tree", "#246d15"), ("Random Forest", "#3f6d15")]):
        preds = results[name]["preds"]
        axes[i].scatter(y_test, preds, alpha=0.3, s=12, color=color)
        mn = min(y_test.min(), preds.min())
        mx = max(y_test.max(), preds.max())
        axes[i].plot([mn, mx], [mn, mx], 'k--', lw=1.5, label="Ligne parfaite")
        axes[i].set_xlabel("Prix réel ($)")
        axes[i].set_ylabel("Prix prédit ($)")
        axes[i].set_title(f"{name} – R²={results[name]['R2']:.3f}")
        axes[i].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
        axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
        axes[i].legend()
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (name, color) in enumerate([("Decision Tree", "#246d15"), ("Random Forest", "#3f6d15")]):
        preds   = results[name]["preds"]
        residus = y_test.values - preds
        axes[i].scatter(preds, residus, alpha=0.3, s=12, color=color)
        axes[i].axhline(0, color='black', lw=1.5, linestyle='--')
        axes[i].set_xlabel("Prix prédit ($)")
        axes[i].set_ylabel("Résidu (réel - prédit)")
        axes[i].set_title(f"Résidus – {name}")
        axes[i].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Des résidus bien centrés autour de 0 indiquent un bon modèle.")

with tab3:
    imp = reg_model.feature_importance()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp["Variable"], imp["Importance"], color="#246d15", alpha=0.85)
    ax.set_title("Importance des variables – Random Forest Régression", fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance (Gini)")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    st.dataframe(imp, use_container_width=True, hide_index=True)

st.divider()

# ── Interprétation ────────────────────────────────────────────────────────────
st.subheader("3 Interprétation")

best = "Random Forest" if results["Random Forest"]["R2"] > results["Decision Tree"]["R2"] else "Decision Tree"
st.markdown(f"""
| Métrique | Interprétation |
|---|---|
| **MAE** | Erreur absolue moyenne en dollars – plus c'est bas, mieux c'est |
| **RMSE** | Pénalise davantage les grosses erreurs – plus c'est bas, mieux c'est |
| **R²** | % de variance expliquée – plus c'est proche de 1, mieux c'est |

**Meilleur modèle** : **{best}** avec R² = {results[best]['R2']:.4f}
""")
