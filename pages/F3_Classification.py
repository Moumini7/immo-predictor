"""
pages/F3_Classification.py – Classification : Type de Bâtiment
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import DataLoader, build_and_train_all, CLF_FEATURES, TARGET_CLF

st.set_page_config(page_title="F3 · Classification", page_icon="🏷️", layout="wide")

from models.model import  Styles_manager

# Charger le CSS
Styles_manager.load_css()


# Titre du dashboard
st.markdown("""
<div class="metric-card metric-card-green">
    <h1 style="margin-bottom: -15px;">
        CLASSIFICATION
    </h1> 
    <span style="margin-right: 10px;"> 
        Comparaison de **SVM** et **Random Forest** pour classifier `BldgType` 
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

with st.spinner("Entraînement des modèles en cours..."):
    prep, reg_model, clf_model, reg_data, clf_data = train()

st.success("Modèles entraînés !")


classes = clf_model.class_names
with st.expander("Variables et classes utilisées"):
    st.markdown("**Variables :** " + ", ".join([f"`{f}`" for f in CLF_FEATURES]))
    st.markdown("**Classes BldgType :** " + " | ".join([f"`{c}`" for c in classes]))

    bldg_desc = {
        "1Fam":   "Maison individuelle (Single-family)",
        "2FmCon": "Conversion deux familles",
        "Duplx":  "Duplex",
        "TwnhsE": "Maison de ville – bout de rangée",
        "TwnhsI": "Maison de ville – intérieure",
    }
    st.table(pd.DataFrame({"Code": list(bldg_desc.keys()), "Description": list(bldg_desc.values())}))

st.divider()

st.subheader("1  Métriques d'évaluation")

results  = clf_data["results"]
y_test   = clf_data["y_test"]

rows = []
for name, m in results.items():
    rows.append({"Modèle": name, "Accuracy": f"{m['Accuracy']:.4f} ({m['Accuracy']*100:.2f}%)", "F1-Score (weighted)": f"{m['F1']:.4f}"})

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("SVM – Accuracy", f"{results['SVM']['Accuracy']*100:.2f}%")
c2.metric("SVM – F1",       f"{results['SVM']['F1']:.4f}")
c3.metric("RF – Accuracy",  f"{results['Random Forest']['Accuracy']*100:.2f}%")
c4.metric("RF – F1",        f"{results['Random Forest']['F1']:.4f}")

st.divider()

st.subheader("2 Visualisations")

tab1, tab2 = st.tabs(["Matrices de confusion", "Rapport de classification"])

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, m) in zip(axes, results.items()):
        cm = m["confusion"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Matrice de confusion – {name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=35, ha='right')
        ax.set_yticklabels(classes)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=11,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("La diagonale = prédictions correctes. Hors diagonale = erreurs de classification.")

with tab2:
    for name, m in results.items():
        st.markdown(f"#### {name}")
        report_df = pd.DataFrame(m["report"]).T
        report_df = report_df.drop(index=["accuracy", "macro avg", "weighted avg"], errors='ignore')
        report_df.index = classes[:len(report_df)]
        report_df = report_df[["precision", "recall", "f1-score", "support"]].round(3)
        report_df.columns = ["Précision", "Rappel", "F1-Score", "Support"]
        st.dataframe(report_df, use_container_width=True)

st.divider()

st.subheader("3 Interprétation")

best = "Random Forest" if results["Random Forest"]["Accuracy"] > results["SVM"]["Accuracy"] else "SVM"
st.markdown(f"""
| Métrique | Interprétation |
|---|---|
| **Accuracy** | % de prédictions correctes sur l'ensemble de test |
| **F1-Score** | Moyenne harmonique précision/rappel – robuste aux classes déséquilibrées |
| **Matrice de confusion** | Détail des vrais/faux positifs et négatifs par classe |

**Meilleur modèle** : **{best}** avec Accuracy = {results[best]['Accuracy']*100:.2f}%
""")
