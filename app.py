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

# Charger le CSS
Styles_manager.load_css()

# Titre du dashboard
st.markdown("""
<div class="metric-card metric-card-green">
    <h1 style="margin-bottom: -15px;">
        TABLEAU DE BORD IMMO PREDICTION
    </h1> 
    <span style="margin-right: 10px;"> 
        Système de Prediction des Immobilier  
    </span>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load():
    return DataLoader.load("data/train.csv")

df = load()


st.subheader("Résumé global du dataset")
summary = DataLoader.summary(df)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Maisons",       summary["n_rows"])
c2.metric("Variables",     summary["n_cols"])
c3.metric("Valeurs manq.", summary["missing_total"])
c4.metric("Prix moyen",   f"{summary['price_mean']:,.0f} $")
c5.metric("Prix médian",  f"{summary['price_median']:,.0f} $")



fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df[TARGET_REG], bins=60, color="#1f8828", edgecolor="white", alpha=0.85)
ax.set_title("Distribution de SalePrice", fontsize=13, fontweight='bold')
ax.set_xlabel("Prix de vente ($)")
ax.set_ylabel("Fréquence")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
st.pyplot(fig)
