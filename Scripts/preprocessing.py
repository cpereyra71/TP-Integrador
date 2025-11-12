
# preprocessing.py
# -*- coding: utf-8 -*-
"""
Preprocesamiento para Properati (memoria optimizada + conversión de moneda a USD)
----------------------------------------------------------------------------------
- Conversión opcional de `price` a USD si `currency` es ARS o UYU (tipo de cambio fijo configurable).
- Imputación de faltantes:
    * Numéricos: mediana
    * Categóricos: más frecuente
- Escalado de numéricos: StandardScaler
- Codificación de categóricos: OneHotEncoder con salida **sparse**
- ColumnTransformer con `sparse_threshold=1.0`
- Guardado de X en .npz si es dispersa, y de y en .npy
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import json
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Variables sugeridas por dominio (si existen en el CSV)
DEFAULT_FEATURES = [
    "rooms", "bedrooms", "bathrooms",
    "surface_total", "surface_covered",
    "lat", "lon",
    "l2", "l3", "property_type",
]

TARGET_COL = "price"  # El target no se transforma aquí, pero se separa de X


@dataclass
class PreprocessArtifacts:
    preprocessor_path: str
    feature_names_path: str
    X_out_path: Optional[str] = None
    y_out_path: Optional[str] = None


def convert_currency_to_usd(df: pd.DataFrame, ars_per_usd: float, uyu_per_usd: float,
                            currency_col: str = "currency", price_col: str = "price") -> pd.DataFrame:
    """Convierte `price` a USD cuando currency es ARS o UYU.
    Fórmula: USD = MONEDA_LOCAL / (LOCAL_por_USD).
    - ARS: usd = price / ars_per_usd
    - UYU: usd = price / uyu_per_usd
    Deja USD como está. Si no hay `currency` o `price`, no hace nada.
    """
    if currency_col not in df.columns or price_col not in df.columns:
        return df

    df = df.copy()
    # Asegurar tipo numérico de price si viniera como string
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # Máscaras
    mask_currency = df[currency_col].astype(str).str.upper()
    mask_ars = mask_currency.eq("ARS")
    mask_uyu = mask_currency.eq("UYU")
    mask_usd = mask_currency.eq("USD")

    # Conversiones
    if ars_per_usd and ars_per_usd > 0:
        df.loc[mask_ars & df[price_col].notna(), price_col] = df.loc[mask_ars & df[price_col].notna(), price_col] / float(ars_per_usd)
    if uyu_per_usd and uyu_per_usd > 0:
        df.loc[mask_uyu & df[price_col].notna(), price_col] = df.loc[mask_uyu & df[price_col].notna(), price_col] / float(uyu_per_usd)

    # Unificar moneda a USD cuando se aplicó conversión o ya era USD
    df.loc[mask_ars | mask_uyu | mask_usd, currency_col] = "USD"

    return df


def _select_features(df: pd.DataFrame,
                     preferred: Optional[List[str]] = None,
                     target: str = TARGET_COL) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    if target not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{target}' en el dataset.")

    y = df[target]
    X = df.drop(columns=[target])

    cols = []
    if preferred:
        cols = [c for c in preferred if c in X.columns]

    if len(cols) < 6:
        auto_numeric = X.select_dtypes(include=["number"]).columns.tolist()
        auto_categ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        blacklist = {"id", "start_date", "end_date", "created_on", "title", "description", "price_period"}
        auto_numeric = [c for c in auto_numeric if c not in blacklist]
        auto_categ = [c for c in auto_categ if c not in blacklist]
        cols = sorted(set(cols + auto_numeric + auto_categ))

    X = X[cols]
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(numeric_features) == 0 and len(categorical_features) == 0:
        raise ValueError("No se detectaron columnas numéricas ni categóricas válidas para preprocesar.")

    return X, y, numeric_features, categorical_features


def _onehot_sparse(dtype=np.float32) -> OneHotEncoder:
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=dtype)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=dtype)


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _onehot_sparse(dtype=np.float32)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=1.0,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer,
                      numeric_features: List[str],
                      categorical_features: List[str]) -> List[str]:
    try:
        out = preprocessor.get_feature_names_out()
        return out.tolist()
    except Exception:
        cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
        return list(numeric_features) + cat_names


def _save_matrix(X_out, path_base: str) -> str:
    if sp.issparse(X_out):
        out_path = f"{path_base}.npz"
        sp.save_npz(out_path, X_out)
    else:
        out_path = f"{path_base}.npy"
        np.save(out_path, X_out)
    return out_path


def fit_transform_and_save(
    df: pd.DataFrame,
    out_dir: str,
    preferred_features: Optional[List[str]] = None,
    target: str = TARGET_COL,
    save_arrays: bool = True,
    convert_currency: bool = True,
    ars_per_usd: float = 850.0,
    uyu_per_usd: float = 40.0,
) -> PreprocessArtifacts:
    if convert_currency:
        df = convert_currency_to_usd(df, ars_per_usd=ars_per_usd, uyu_per_usd=uyu_per_usd)

    X, y, num_cols, cat_cols = _select_features(df, preferred_features, target=target)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_out = preprocessor.fit_transform(X)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preprocessor_path = str(out / "preprocessor.joblib")
    feature_names_path = str(out / "feature_names.json")
    X_out_base = str(out / "X_preprocessed")
    y_out_path = str(out / "y.npy") if save_arrays else None

    joblib.dump(preprocessor, preprocessor_path)

    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)
    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    X_out_path = None
    if save_arrays:
        if y.isna().any():
            mask = ~y.isna()
            X_filtered = X_out[mask.values] if sp.issparse(X_out) else X_out[mask.values, :]
            X_out_path = _save_matrix(X_filtered, X_out_base + "_aligned")
            np.save(y_out_path, y[mask].to_numpy())
        else:
            X_out_path = _save_matrix(X_out, X_out_base)
            np.save(y_out_path, y.to_numpy())

    return PreprocessArtifacts(
        preprocessor_path=preprocessor_path,
        feature_names_path=feature_names_path,
        X_out_path=X_out_path,
        y_out_path=y_out_path,
    )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Preprocesamiento Properati (memoria + conversión moneda)")
    parser.add_argument("--csv", type=str, default="properati_clean.csv", help="Ruta al CSV de entrada")
    parser.add_argument("--out_dir", type=str, default="artifacts", help="Directorio de salida")
    parser.add_argument("--no_arrays", action="store_true", help="No guardar matrices")
    parser.add_argument("--no_convert_currency", action="store_true", help="Desactiva conversión ARS/UYU -> USD")
    parser.add_argument("--ars_usd", type=float, default=850.0, help="Pesos argentinos por 1 USD (fijo)")
    parser.add_argument("--uyu_usd", type=float, default=40.0, help="Pesos uruguayos por 1 USD (fijo)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"No se encontró el archivo {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    artifacts = fit_transform_and_save(
        df=df,
        out_dir=args.out_dir,
        preferred_features=DEFAULT_FEATURES,
        save_arrays=not args.no_arrays,
        convert_currency=not args.no_convert_currency,
        ars_per_usd=args.ars_usd,
        uyu_per_usd=args.uyu_usd,
    )

    print("Preprocesamiento completado.")
    print(f"Preprocessor: {artifacts.preprocessor_path}")
    print(f"Feature names: {artifacts.feature_names_path}")
    if artifacts.X_out_path:
        print(f"X preprocesado: {artifacts.X_out_path}")
    if artifacts.y_out_path:
        print(f"y: {artifacts.y_out_path}")
