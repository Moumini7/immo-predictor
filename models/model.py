import pandas as pd
import numpy as np
import os
import streamlit as st
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# CONSTANTES : variables utilisées par chaque tâche
# ─────────────────────────────────────────────

REG_FEATURES = [
    'GrLivArea', 'TotalBsmtSF', 'LotArea', 'BedroomAbvGr',
    'FullBath', 'TotRmsAbvGrd', 'OverallQual', 'OverallCond',
    'YearBuilt', 'YearRemodAdd', 'Neighborhood',
    'GarageCars', 'GarageArea', 'PoolArea', 'Fireplaces'
]

CLF_FEATURES = [
    'GrLivArea', 'TotRmsAbvGrd', 'OverallQual',
    'YearBuilt', 'GarageCars', 'Neighborhood', 'HouseStyle'
]

TARGET_REG = 'SalePrice'
TARGET_CLF = 'BldgType'

NEIGHBORHOOD_LIST = [
    'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
    'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
    'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown',
    'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'
]

HOUSESTYLE_LIST = [
    '1Story', '1.5Fin', '1.5Unf', '2Story',
    '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'
]

BLDGTYPE_LIST = ['1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI']


# ─────────────────────────────────────────────
# CLASSE 1 : DataLoader
# ─────────────────────────────────────────────

class DataLoader:
    """Charge et met en cache le dataset."""

    @staticmethod
    @st.cache_data
    def load(path: str = "data/train.csv") -> pd.DataFrame:
        """Charge le CSV et retourne un DataFrame."""
        df = pd.read_csv(path)
        return df

    @staticmethod
    def summary(df: pd.DataFrame) -> dict:
        """Retourne un résumé rapide du dataset."""
        return {
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "missing_total": int(df.isnull().sum().sum()),
            "missing_cols": int((df.isnull().sum() > 0).sum()),
            "numeric_cols": int(df.select_dtypes(include=[np.number]).shape[1]),
            "categoric_cols": int(df.select_dtypes(include=['object']).shape[1]),
            "price_mean": float(df[TARGET_REG].mean()),
            "price_median": float(df[TARGET_REG].median()),
            "price_std": float(df[TARGET_REG].std()),
            "price_min": float(df[TARGET_REG].min()),
            "price_max": float(df[TARGET_REG].max()),
        }

    @staticmethod
    def missing_report(df: pd.DataFrame) -> pd.DataFrame:
        """Retourne un DataFrame des colonnes avec valeurs manquantes."""
        miss = df.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        pct = (miss / len(df) * 100).round(2)
        return pd.DataFrame({"Manquants": miss, "Pourcentage (%)": pct})


# ─────────────────────────────────────────────
# CLASSE 2 : Preprocessor
# ─────────────────────────────────────────────

class Preprocessor:
    """
    Gère :
    - les valeurs manquantes
    - l'encodage des variables catégorielles
    - la standardisation
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_encoder_clf = LabelEncoder()

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    def _encode_categoricals(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        df = df.copy()
        for col in cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Gère les valeurs inconnues lors du predict
                known = list(self.label_encoders[col].classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else known[0]
                )
                df[col] = self.label_encoders[col].transform(df[col])
        return df

    def prepare_regression(self, df: pd.DataFrame):
        """
        Prépare X et y pour la régression.
        Retourne X_train, X_test, y_train, y_test
        """
        data = df[REG_FEATURES + [TARGET_REG]].copy()
        data = self._fill_missing(data)

        cat_cols = data[REG_FEATURES].select_dtypes(include=['object']).columns.tolist()
        data = self._encode_categoricals(data, cat_cols)

        X = data[REG_FEATURES]
        y = data[TARGET_REG]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def prepare_classification(self, df: pd.DataFrame):
        """
        Prépare X et y pour la classification.
        Retourne X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
        """
        data = df[CLF_FEATURES + [TARGET_CLF]].copy()
        data = self._fill_missing(data)

        cat_cols = data[CLF_FEATURES].select_dtypes(include=['object']).columns.tolist()
        data = self._encode_categoricals(data, cat_cols)

        # Encodage de la cible
        data[TARGET_CLF] = self.target_encoder_clf.fit_transform(data[TARGET_CLF].astype(str))

        X = data[CLF_FEATURES]
        y = data[TARGET_CLF]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, X_train_s, X_test_s

    def encode_single_reg(self, input_dict: dict) -> pd.DataFrame:
        """Encode un seul enregistrement pour la prédiction régression."""
        row = pd.DataFrame([input_dict])
        cat_cols = row.select_dtypes(include=['object']).columns.tolist()
        row = self._encode_categoricals(row, cat_cols)
        return row[REG_FEATURES]

    def encode_single_clf(self, input_dict: dict) -> tuple:
        """Encode un seul enregistrement pour la prédiction classification."""
        row = pd.DataFrame([input_dict])
        cat_cols = row.select_dtypes(include=['object']).columns.tolist()
        row = self._encode_categoricals(row, cat_cols)
        X = row[CLF_FEATURES]
        X_scaled = self.scaler.transform(X)
        return X, X_scaled


# ─────────────────────────────────────────────
# CLASSE 3 : RegressionModels
# ─────────────────────────────────────────────

class RegressionModels:
    """Entraîne et évalue Decision Tree et Random Forest pour la régression."""

    def __init__(self):
        self.decision_tree = DecisionTreeRegressor(max_depth=8, random_state=42)
        self.random_forest = RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        )
        self._is_trained = False

    def train(self, X_train, y_train):
        self.decision_tree.fit(X_train, y_train)
        self.random_forest.fit(X_train, y_train)
        self._is_trained = True

    def evaluate(self, X_test, y_test) -> dict:
        """Retourne les métriques MAE, RMSE, R² pour les 2 modèles."""
        results = {}
        for name, model in [("Decision Tree", self.decision_tree),
                             ("Random Forest", self.random_forest)]:
            preds = model.predict(X_test)
            results[name] = {
                "preds": preds,
                "MAE":   round(mean_absolute_error(y_test, preds), 2),
                "RMSE":  round(np.sqrt(mean_squared_error(y_test, preds)), 2),
                "R2":    round(r2_score(y_test, preds), 4),
            }
        return results

    def feature_importance(self) -> pd.DataFrame:
        """Retourne l'importance des variables du Random Forest."""
        return pd.DataFrame({
            "Variable":   REG_FEATURES,
            "Importance": self.random_forest.feature_importances_
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

    def predict_price(self, X_encoded: pd.DataFrame) -> dict:
        """Prédit le prix avec les 2 modèles."""
        return {
            "Decision Tree": float(self.decision_tree.predict(X_encoded)[0]),
            "Random Forest": float(self.random_forest.predict(X_encoded)[0]),
        }


# ─────────────────────────────────────────────
# CLASSE 4 : ClassificationModels
# ─────────────────────────────────────────────

class ClassificationModels:
    """Entraîne et évalue SVM et Random Forest pour la classification."""

    def __init__(self):
        self.svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        self.random_forest = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        )
        self._is_trained = False
        self.class_names = []

    def train(self, X_train, y_train, X_train_scaled, class_names: list):
        self.svm.fit(X_train_scaled, y_train)
        self.random_forest.fit(X_train, y_train)
        self.class_names = class_names
        self._is_trained = True

    def evaluate(self, X_test, y_test, X_test_scaled) -> dict:
        """Retourne Accuracy, F1-score et matrice de confusion pour les 2 modèles."""
        results = {}
        for name, model, X in [("SVM", self.svm, X_test_scaled),
                                 ("Random Forest", self.random_forest, X_test)]:
            preds = model.predict(X)
            results[name] = {
                "preds":       preds,
                "Accuracy":    round(accuracy_score(y_test, preds), 4),
                "F1":          round(f1_score(y_test, preds, average='weighted'), 4),
                "confusion":   confusion_matrix(y_test, preds),
                "report":      classification_report(
                    y_test, preds,
                    target_names=self.class_names,
                    output_dict=True
                ),
            }
        return results

    def feature_importance(self) -> pd.DataFrame:
        """Retourne l'importance des variables du Random Forest Classifier."""
        return pd.DataFrame({
            "Variable":   CLF_FEATURES,
            "Importance": self.random_forest.feature_importances_
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

    def predict_type(self, X, X_scaled, target_encoder) -> dict:
        """Prédit le type de bâtiment avec les 2 modèles."""
        pred_svm = self.svm.predict(X_scaled)[0]
        pred_rf  = self.random_forest.predict(X)[0]
        return {
            "SVM":          target_encoder.inverse_transform([pred_svm])[0],
            "Random Forest": target_encoder.inverse_transform([pred_rf])[0],
        }


# ─────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────

def get_correlation_with_price(df: pd.DataFrame) -> pd.Series:
    """Corrélation de toutes les variables numériques avec SalePrice."""
    num_df = df[REG_FEATURES + [TARGET_REG]].select_dtypes(include=[np.number])
    return num_df.corr()[TARGET_REG].drop(TARGET_REG).sort_values(ascending=False)


def build_and_train_all(df: pd.DataFrame):
    """
    Fonction principale : prépare les données, entraîne tous les modèles.
    Retourne (preprocessor, reg_models, clf_models, reg_data, clf_data)
    Mis en cache avec st.cache_resource pour ne pas ré-entraîner à chaque clic.
    """
    prep = Preprocessor()

    # Régression
    Xtr_r, Xte_r, ytr_r, yte_r = prep.prepare_regression(df)
    reg = RegressionModels()
    reg.train(Xtr_r, ytr_r)
    reg_results = reg.evaluate(Xte_r, yte_r)

    # Classification
    Xtr_c, Xte_c, ytr_c, yte_c, Xtr_cs, Xte_cs = prep.prepare_classification(df)
    clf = ClassificationModels()
    clf.train(Xtr_c, ytr_c, Xtr_cs, list(prep.target_encoder_clf.classes_))
    clf_results = clf.evaluate(Xte_c, yte_c, Xte_cs)

    return (
        prep, reg, clf,
        {"X_train": Xtr_r, "X_test": Xte_r, "y_train": ytr_r, "y_test": yte_r, "results": reg_results},
        {"X_train": Xtr_c, "X_test": Xte_c, "y_train": ytr_c, "y_test": yte_c,
         "X_train_s": Xtr_cs, "X_test_s": Xte_cs, "results": clf_results}
    )


class Styles_manager:

    @staticmethod
    def load_css():
        # 1. Configuration de la page (Doit être appelé une seule fois dans l'app)
        try:
            st.set_page_config(
                page_title="Système de Gestion de collecte de Données",
                layout="wide",
                page_icon="🧊"
            )
        except st.errors.StreamlitAPIException:
            # Si déjà configuré ailleurs, on ignore pour éviter le crash
            pass

        # 2. Gestion des chemins absolus
        # __file__ est dans /models/model.py, donc .parent.parent est la racine du projet
        current_dir = Path(__file__).resolve().parent.parent
        
        # Chemin vers le logo
        LOGO_PATH = str(current_dir / "images" / "sygescol_ltd.png")
        
        # 3. Chargement du Logo avec vérification de présence
        if os.path.exists(LOGO_PATH):
            st.logo(LOGO_PATH, icon_image=LOGO_PATH, size="large")
        else:
            st.sidebar.warning(f"Logo non trouvé : {LOGO_PATH}")

        # 4. Chargement des Font Awesome et CSS
        st.markdown("""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
            """, unsafe_allow_html=True)
        
        css_file = current_dir / "utils" / "styles.css"
        if css_file.exists():
            with open(css_file, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
