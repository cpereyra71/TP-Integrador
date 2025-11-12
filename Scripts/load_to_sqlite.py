
# load_to_sqlite.py
# -*- coding: utf-8 -*-
"""
Carga de datos a SQLite para el TP Properati
============================================
- Lee el CSV limpio (`properati_clean.csv`).
- Opcional: aplica la misma conversión de moneda (ARS/UYU -> USD) que el preprocesamiento.
- Lee artefactos del preprocesamiento (preprocessor.joblib, feature_names.json y X_preprocessed.*).
- Crea una base SQLite con 3 tablas:
    * input_data: datos de entrada (X crudo seleccionado + y)
    * preprocessed_data: matriz preprocesada en formato disperso (row_idx, feature, value)
    * model_config: metadatos de configuración (paths, TC usado, etc.)
    * model_results: (vacía al inicio) para predicciones y métricas
Uso:
    python load_to_sqlite.py --csv properati_clean.csv --artifacts_dir artifacts --db_path artifacts/database.db --store both
"""

from __future__ import annotations
import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

try:
    import scipy.sparse as sp
except Exception:
    sp = None


def _load_preprocessed_matrix(artifacts_dir: Path):
    npz = artifacts_dir / "X_preprocessed.npz"
    npz_aligned = artifacts_dir / "X_preprocessed_aligned.npz"
    npy = artifacts_dir / "X_preprocessed.npy"
    npy_aligned = artifacts_dir / "X_preprocessed_aligned.npy"

    if npz.exists():
        if sp is None:
            raise RuntimeError("Se encontró X_preprocessed.npz pero SciPy no está disponible para cargar sparse.")
        return sp.load_npz(npz), "sparse"
    if npz_aligned.exists():
        if sp is None:
            raise RuntimeError("Se encontró X_preprocessed_aligned.npz pero SciPy no está disponible para cargar sparse.")
        return sp.load_npz(npz_aligned), "sparse"
    if npy.exists():
        return np.load(npy), "dense"
    if npy_aligned.exists():
        return np.load(npy_aligned), "dense"
    raise FileNotFoundError("No se encontró X_preprocessed(.npz/.npy) en artifacts_dir.")


def _iter_sparse_triplets(X):
    if sp is None:
        raise RuntimeError("SciPy no disponible para iterar sparse.")
    if sp.isspmatrix_coo(X):
        coo = X
    else:
        coo = X.tocoo()
    for r, c, v in zip(coo.row, coo.col, coo.data):
        yield int(r), int(c), float(v)


def convert_currency_to_usd(df: pd.DataFrame, ars_per_usd: float, uyu_per_usd: float,
                            currency_col: str = "currency", price_col: str = "price") -> pd.DataFrame:
    if currency_col not in df.columns or price_col not in df.columns:
        return df
    df = df.copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    cur = df[currency_col].astype(str).str.upper()
    mask_ars = cur.eq("ARS")
    mask_uyu = cur.eq("UYU")
    mask_usd = cur.eq("USD")
    if ars_per_usd and ars_per_usd > 0:
        df.loc[mask_ars & df[price_col].notna(), price_col] = df.loc[mask_ars & df[price_col].notna(), price_col] / float(ars_per_usd)
    if uyu_per_usd and uyu_per_usd > 0:
        df.loc[mask_uyu & df[price_col].notna(), price_col] = df.loc[mask_uyu & df[price_col].notna(), price_col] / float(uyu_per_usd)
    df.loc[mask_ars | mask_uyu | mask_usd, currency_col] = "USD"
    return df


def create_tables(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS input_data (
            row_idx INTEGER PRIMARY KEY
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS preprocessed_data (
            row_idx INTEGER NOT NULL,
            feature TEXT NOT NULL,
            value REAL NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            row_idx INTEGER,
            y_true REAL,
            y_pred REAL,
            split TEXT,
            model_name TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            preprocessor_path TEXT,
            feature_names_path TEXT,
            ars_usd REAL,
            uyu_usd REAL,
            notes TEXT
        );
    """)
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Carga datos/artefactos a SQLite")
    parser.add_argument("--csv", type=str, default="properati_clean.csv")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--db_path", type=str, default="artifacts/database.db")
    parser.add_argument("--store", type=str, choices=["raw", "pre", "both"], default="both")
    parser.add_argument("--convert_currency", action="store_true")
    parser.add_argument("--ars_usd", type=float, default=850.0)
    parser.add_argument("--uyu_usd", type=float, default=40.0)
    args = parser.parse_args()

    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        raise SystemExit(f"No existe artifacts_dir: {artifacts_dir.resolve()}")

    preproc_path = artifacts_dir / "preprocessor.joblib"
    if not preproc_path.exists():
        raise SystemExit(f"Falta preprocessor.joblib en {artifacts_dir}")
    preprocessor = joblib.load(preproc_path)

    featnames_path = artifacts_dir / "feature_names.json"
    if not featnames_path.exists():
        raise SystemExit(f"Falta feature_names.json en {artifacts_dir}")
    feature_names = json.loads(Path(featnames_path).read_text(encoding="utf-8"))

    num_cols = list(preprocessor.transformers_[0][2]) if preprocessor.transformers_ else []
    cat_cols = list(preprocessor.transformers_[1][2]) if len(preprocessor.transformers_) > 1 else []
    raw_cols = num_cols + cat_cols

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        create_tables(conn)

        if args.store in ("raw", "both"):
            df = pd.read_csv(args.csv)
            if args.convert_currency:
                df = convert_currency_to_usd(df, ars_per_usd=args.ars_usd, uyu_per_usd=args.uyu_usd)
            keep = [c for c in raw_cols if c in df.columns]
            subset = df[keep].copy()
            if "price" in df.columns:
                subset["price"] = pd.to_numeric(df["price"], errors="coerce")
            subset.reset_index(drop=True, inplace=True)
            subset.insert(0, "row_idx", subset.index.astype(int))
            subset.to_sql("input_data", conn, if_exists="replace", index=False)

        if args.store in ("pre", "both"):
            X, fmt = _load_preprocessed_matrix(artifacts_dir)
            if fmt == "dense":
                if sp is None:
                    raise RuntimeError("SciPy no disponible para manejar formato disperso.")
                X = sp.csr_matrix(X)

            cur = conn.cursor()
            cur.execute("DELETE FROM preprocessed_data;")

            feat_by_idx = {i: name for i, name in enumerate(feature_names)}

            batch = []
            BATCH_SIZE = 200000
            # iterate sparse triplets
            if sp is None:
                raise RuntimeError("SciPy no disponible para iterar sparse.")
            coo = X if sp.isspmatrix_coo(X) else X.tocoo()
            for r, c, v in zip(coo.row, coo.col, coo.data):
                batch.append((int(r), feat_by_idx.get(int(c), f"f{int(c)}"), float(v)))
                if len(batch) >= BATCH_SIZE:
                    cur.executemany("INSERT INTO preprocessed_data (row_idx, feature, value) VALUES (?, ?, ?);", batch)
                    conn.commit()
                    batch = []
            if batch:
                cur.executemany("INSERT INTO preprocessed_data (row_idx, feature, value) VALUES (?, ?, ?);", batch)
                conn.commit()

        cur = conn.cursor()
        cur.execute(
            "INSERT INTO model_config (run_id, preprocessor_path, feature_names_path, ars_usd, uyu_usd, notes) VALUES (?, ?, ?, ?, ?, ?);",
            (run_id, str(preproc_path), str(featnames_path), args.ars_usd, args.uyu_usd,
             "Carga inicial de datos/artefactos")
        )
        conn.commit()

        print(f"Base SQLite creada en: {db_path}")
        print("Tablas: input_data, preprocessed_data, model_results, model_config")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
