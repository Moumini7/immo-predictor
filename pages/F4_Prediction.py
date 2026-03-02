"""
pages/F4_Prediction.py – Prédiction Interactive
L'utilisateur entre les caractéristiques d'une maison
→ prix estimé + type de bâtiment prédit
"""

import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import (
    DataLoader, build_and_train_all,
    REG_FEATURES, CLF_FEATURES,
    NEIGHBORHOOD_LIST, HOUSESTYLE_LIST, BLDGTYPE_LIST
)

from models.model import  Styles_manager

# Charger le CSS
Styles_manager.load_css()


# Titre du dashboard
st.markdown("""
<div class="metric-card metric-card-green">
    <h1 style="margin-bottom: -15px;">
        PREDICTION
    </h1> 
    <span style="margin-right: 10px;"> 
        Prédiction Interactive 
    </span>
</div>
""", unsafe_allow_html=True)
st.divider()


# ── Chargement et entraînement (mis en cache) ─────────────────────────────────
@st.cache_resource
def train():
    df = DataLoader.load("data/train.csv")
    return build_and_train_all(df)

with st.spinner("Chargement des modèles..."):
    prep, reg_model, clf_model, reg_data, clf_data = train()

# ── Formulaire de saisie ──────────────────────────────────────────────────────
st.subheader("Caractéristiques de la maison")

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Surfaces (sq ft)**")
        GrLivArea    = st.number_input("Surface habitable (GrLivArea)",    100, 6000, 1500, step=50)
        TotalBsmtSF  = st.number_input("Surface sous-sol (TotalBsmtSF)",     0, 6000, 800,  step=50)
        LotArea      = st.number_input("Surface du terrain (LotArea)",     1000, 100000, 8000, step=500)
        GarageArea   = st.number_input("Surface garage (GarageArea)",        0, 1500, 480, step=10)
        PoolArea     = st.number_input("Surface piscine (PoolArea)",          0, 800,  0,   step=10)

    with col2:
        st.markdown("**Caractéristiques**")
        OverallQual  = st.slider("Qualité générale (OverallQual)",  1, 10, 6)
        OverallCond  = st.slider("Condition générale (OverallCond)", 1, 10, 5)
        BedroomAbvGr = st.number_input("Chambres (BedroomAbvGr)",   0, 10, 3)
        FullBath     = st.number_input("Salles de bain complètes (FullBath)", 0, 5, 2)
        TotRmsAbvGrd = st.number_input("Total pièces (TotRmsAbvGrd)", 1, 15, 7)
        GarageCars   = st.selectbox("Places de garage (GarageCars)", [0, 1, 2, 3, 4])
        Fireplaces   = st.number_input("Cheminées (Fireplaces)", 0, 5, 0)

    with col3:
        st.markdown("**Année & Localisation**")
        YearBuilt    = st.number_input("Année de construction (YearBuilt)", 1870, 2024, 1990)
        YearRemodAdd = st.number_input("Année rénovation (YearRemodAdd)", 1870, 2024, 1990)
        Neighborhood = st.selectbox("Quartier (Neighborhood)", NEIGHBORHOOD_LIST)
        HouseStyle   = st.selectbox("Style de maison (HouseStyle)", HOUSESTYLE_LIST)

    st.divider()

    submitted = st.form_submit_button("Prédire", type="primary", use_container_width=True)

if submitted:
    # Dictionnaire des inputs
    input_reg = {
        "GrLivArea": GrLivArea, "TotalBsmtSF": TotalBsmtSF, "LotArea": LotArea,
        "BedroomAbvGr": BedroomAbvGr, "FullBath": FullBath, "TotRmsAbvGrd": TotRmsAbvGrd,
        "OverallQual": OverallQual, "OverallCond": OverallCond,
        "YearBuilt": YearBuilt, "YearRemodAdd": YearRemodAdd,
        "Neighborhood": Neighborhood, "GarageCars": GarageCars,
        "GarageArea": GarageArea, "PoolArea": PoolArea, "Fireplaces": Fireplaces,
    }
    input_clf = {
        "GrLivArea": GrLivArea, "TotRmsAbvGrd": TotRmsAbvGrd,
        "OverallQual": OverallQual, "YearBuilt": YearBuilt,
        "GarageCars": GarageCars, "Neighborhood": Neighborhood, "HouseStyle": HouseStyle,
    }

    try:
        # Encodage
        X_reg             = prep.encode_single_reg(input_reg)
        X_clf, X_clf_s    = prep.encode_single_clf(input_clf)

        # Prédictions
        price_preds = reg_model.predict_price(X_reg)
        type_preds  = clf_model.predict_type(X_clf, X_clf_s, prep.target_encoder_clf)

        st.divider()
        st.subheader("Résultats de la prédiction")

        # ── Prix ────────────────────────────────────────────────────────────
        st.markdown("### Prix de vente estimé")
        c1, c2, c3 = st.columns(3)

        dt_price = price_preds["Decision Tree"]
        rf_price = price_preds["Random Forest"]
        avg_price = (dt_price + rf_price) / 2

        c1.metric("Decision Tree",  f"{dt_price:,.0f} $")
        c2.metric("Random Forest",  f"{rf_price:,.0f} $")
        c3.metric("Moyenne",     f"{avg_price:,.0f} $",
                  help="Moyenne des 2 modèles – estimation consensuelle")

        # Jauge de prix
        df_ref = DataLoader.load("data/train.csv")
        pmin, pmax = df_ref["SalePrice"].min(), df_ref["SalePrice"].max()
        pct = (avg_price - pmin) / (pmax - pmin) * 100

        st.markdown(f"**Position dans le marché :** Le prix estimé se situe dans le **{pct:.0f}e percentile** des prix du dataset.")
        st.progress(min(int(pct), 100))

        quartiles = df_ref["SalePrice"].quantile([0.25, 0.5, 0.75])
        if avg_price < quartiles[0.25]:
            st.info("🟢 Bien **en dessous** de la médiane du marché – prix attractif.")
        elif avg_price < quartiles[0.5]:
            st.info("🟡 **En dessous** de la médiane du marché.")
        elif avg_price < quartiles[0.75]:
            st.warning("🟠 **Au-dessus** de la médiane du marché.")
        else:
            st.error("🔴 Dans le **top 25%** des prix – bien haut de gamme.")

        # ── Type de bâtiment ────────────────────────────────────────────────
        st.markdown("### Type de bâtiment prédit")
        c1, c2 = st.columns(2)

        bldg_desc = {
            "1Fam":   "Maison individuelle (Single-family Detached)",
            "2FmCon": "Conversion deux familles",
            "Duplx":  "Duplex",
            "TwnhsE": "Maison de ville – bout de rangée (End Unit)",
            "TwnhsI": "Maison de ville – intérieure (Inside Unit)",
        }

        c1.metric("SVM",          type_preds["SVM"])
        c1.caption(bldg_desc.get(type_preds["SVM"], ""))
        c2.metric("Random Forest", type_preds["Random Forest"])
        c2.caption(bldg_desc.get(type_preds["Random Forest"], ""))

        if type_preds["SVM"] == type_preds["Random Forest"]:
            st.success(f"Les deux modèles sont **d'accord** : `{type_preds['Random Forest']}`")
        else:
            st.warning("Les deux modèles sont en **désaccord**. Faites confiance au Random Forest (meilleure performance).")

        # ── Récapitulatif des inputs ────────────────────────────────────────
        with st.expander("Récapitulatif des caractéristiques saisies"):
            recap = pd.DataFrame({
                "Variable": list(input_reg.keys()),
                "Valeur":   list(input_reg.values())
            })
            st.dataframe(recap, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        st.exception(e)