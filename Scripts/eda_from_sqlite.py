# Proyecto TPPA
# Scripts/eda_from_sqlite.py
# -*- coding: utf-8 -*-
"""
EDA básico desde SQLite (simple y claro)
- Lee la tabla 'input_data' desde artifacts/database.db (configurable por CLI).
- Genera resúmenes: shape, dtypes, nulos, describe numérico, top categorías.
- Crea gráficos con matplotlib (sin seaborn, gráficos individuales).
- Guarda todo en la carpeta indicada (por defecto artifacts/eda/).
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------- Utilidades de guardado --------------------
def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] Guardado CSV: {path}")


def save_txt(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[OK] Guardado TXT: {path}")


def safe_cols(df: pd.DataFrame, cols):
    """Devuelve solo las columnas que existan en el DataFrame."""
    return [c for c in cols if c in df.columns]


# -------------------- Gráficos (matplotlib puro) --------------------
def plot_hist(series: pd.Series, title: str, out_path: Path, bins=50, logx=False):
    data = series.dropna().to_numpy()
    if len(data) == 0:
        print(f"[WARN] {series.name}: no hay datos para histograma.")
        return
    plt.figure()
    if logx:
        data = data[data > 0]
        if len(data) == 0:
            print(f"[WARN] {series.name}: no hay datos > 0 para log10.")
            plt.close()
            return
        data = np.log10(data)
        plt.hist(data, bins=bins)
        plt.xlabel("log10(valor)")
    else:
        plt.hist(data, bins=bins)
        plt.xlabel(series.name)
    plt.ylabel("Frecuencia")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[OK] Gráfico: {out_path}")


def plot_scatter(x: pd.Series, y: pd.Series, title: str, out_path: Path, sample=5000):
    dx = x.copy()
    dy = y.copy()

    # Alinear índices y muestrear si hay muchas filas
    common_idx = dx.dropna().index.intersection(dy.dropna().index)
    if len(common_idx) == 0:
        print(f"[WARN] {x.name} vs {y.name}: no hay puntos comunes.")
        return
    if len(common_idx) > sample:
        common_idx = np.random.choice(common_idx, size=sample, replace=False)

    dx = dx.loc[common_idx]
    dy = dy.loc[common_idx]

    plt.figure()
    plt.scatter(dx, dy, s=5, alpha=0.5)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[OK] Gráfico: {out_path}")


def plot_bar(series: pd.Series, title: str, out_path: Path, top=15):
    s = series.dropna().sort_values(ascending=False).head(top)
    if s.empty:
        print(f"[WARN] Serie vacía para barras: {title}")
        return
    plt.figure()
    s.plot(kind="bar")
    plt.title(title)
    plt.ylabel("Valor")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[OK] Gráfico: {out_path}")


def plot_box_by_category(df: pd.DataFrame, value_col: str, cat_col: str,
                         title: str, out_path: Path, top=8):
    if cat_col not in df.columns or value_col not in df.columns:
        print(f"[WARN] Faltan columnas para boxplot {value_col} por {cat_col}")
        return
    tops = df[cat_col].value_counts().head(top).index.tolist()
    if len(tops) == 0:
        print(f"[WARN] Sin categorías para boxplot en {cat_col}")
        return
    data = [df.loc[df[cat_col] == t, value_col].dropna().to_numpy() for t in tops]
    if all(len(arr) == 0 for arr in data):
        print(f"[WARN] Sin datos numéricos para boxplot {value_col} por {cat_col}")
        return
    plt.figure()
    plt.boxplot(data, labels=tops, showfliers=False)
    plt.title(title)
    plt.ylabel(value_col)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[OK] Gráfico: {out_path}")


def plot_corr_heatmap(df_num: pd.DataFrame, title: str, out_path: Path):
    if df_num.shape[1] < 2:
        print("[WARN] Correlación: se necesitan >= 2 columnas numéricas.")
        return
    plt.figure()
    corr = df_num.select_dtypes(include=[np.number]).corr()
    im = plt.imshow(corr.values, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[OK] Gráfico: {out_path}")


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="EDA básico desde SQLite")
    parser.add_argument("--db_path", type=str, default="artifacts/database.db",
                        help="Ruta a la base SQLite")
    parser.add_argument("--table", type=str, default="input_data",
                        help="Nombre de la tabla a leer")
    parser.add_argument("--out_dir", type=str, default="artifacts/eda",
                        help="Directorio de salida para CSV/PNG")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise SystemExit(f"No se encontró la base de datos: {db_path.resolve()}")

    print(f"Conectando a SQLite: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(f"SELECT * FROM {args.table};", conn)
    finally:
        conn.close()

    print(f"Filas: {len(df)}, Columnas: {len(df.columns)}")

    # 1) Información básica
    save_txt(f"shape = {df.shape}", out_dir / "01_shape.txt")

    dtypes = df.dtypes.reset_index()
    dtypes.columns = ["column", "dtype"]
    save_csv(dtypes, out_dir / "02_dtypes.csv")

    nulls = df.isna().sum().reset_index()
    nulls.columns = ["column", "n_nulls"]
    nulls["pct_nulls"] = (nulls["n_nulls"] / max(len(df), 1) * 100).round(2)
    save_csv(nulls, out_dir / "03_nulls.csv")

    desc = df.describe(include=[np.number]).T.reset_index()
    desc.rename(columns={"index": "column"}, inplace=True)
    save_csv(desc, out_dir / "04_describe_numeric.csv")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        top_list = []
        for c in cat_cols:
            vc = df[c].value_counts(dropna=False).head(20).reset_index()
            vc.columns = [c, "count"]
            vc.insert(0, "column", c)
            top_list.append(vc)
        top_df = pd.concat(top_list, ignore_index=True)
        save_csv(top_df, out_dir / "05_top_categories.csv")

    # 2) Gráficos sugeridos (solo si existen columnas)
    num_candidates = ["price", "surface_total", "surface_covered", "rooms", "bedrooms", "bathrooms"]
    cat_candidates = ["property_type", "l2", "l3"]
    geo_candidates = ["lat", "lon"]

    num_cols_present = safe_cols(df, num_candidates)
    cat_cols_present = safe_cols(df, cat_candidates)
    geo_cols_present = safe_cols(df, geo_candidates)

    # Histograma de price
    if "price" in df.columns:
        plot_hist(df["price"], "Histograma de price (USD)",
                  out_dir / "A1_hist_price.png", bins=60, logx=False)
        plot_hist(df["price"], "Histograma de price (log10 USD)",
                  out_dir / "A2_hist_price_log.png", bins=60, logx=True)

    # Barras: precio promedio por property_type
    if "property_type" in df.columns and "price" in df.columns:
        avg_price_by_pt = df.groupby("property_type")["price"].mean().sort_values(ascending=False).round(2)
        save_csv(avg_price_by_pt.reset_index().rename(columns={"price": "avg_price_usd"}),
                 out_dir / "B1_avg_price_by_property_type.csv")
        plot_bar(avg_price_by_pt, "Precio promedio por property_type (USD)",
                 out_dir / "B1_avg_price_by_property_type.png", top=15)

    # Boxplot: price por property_type
    if "property_type" in df.columns and "price" in df.columns:
        plot_box_by_category(df, "price", "property_type",
                             "Distribución de price por property_type",
                             out_dir / "B2_box_price_by_property_type.png", top=8)

    # Scatter: superficie_total vs price
    if "surface_total" in df.columns and "price" in df.columns:
        plot_scatter(df["surface_total"], df["price"],
                     "surface_total vs price",
                     out_dir / "C1_scatter_surface_total_price.png",
                     sample=8000)

    # Correlaciones numéricas (con las que estén disponibles)
    if num_cols_present:
        plot_corr_heatmap(df[num_cols_present],
                          "Matriz de correlaciones (numéricas)",
                          out_dir / "D1_corr_heatmap.png")

    # Mapa simple lat/lon coloreado por price
    if all(c in df.columns for c in ["lat", "lon", "price"]):
        idx = np.arange(len(df))
        if len(idx) > 10000:
            idx = np.random.choice(idx, size=10000, replace=False)
        plt.figure()
        plt.scatter(df.loc[idx, "lon"], df.loc[idx, "lat"], c=df.loc[idx, "price"], s=3)
        plt.xlabel("lon")
        plt.ylabel("lat")
        plt.title("Distribución geográfica (color = price)")
        plt.tight_layout()
        plt.savefig(out_dir / "E1_geo_scatter_price.png", dpi=120)
        plt.close()
        print(f"[OK] Gráfico: {out_dir / 'E1_geo_scatter_price.png'}")

    # 3) Resumen final
    lines = [
        f"Filas = {len(df)}, Columnas = {len(df.columns)}",
        "Numéricas consideradas: " + (", ".join(num_cols_present) if num_cols_present else "—"),
        "Categóricas consideradas: " + (", ".join(cat_cols_present) if cat_cols_present else "—"),
        "Geográficas disponibles: " + (", ".join(geo_cols_present) if geo_cols_present else "—")
    ]
    save_txt("\n".join(lines), out_dir / "00_resumen_eda.txt")

    print("\n[OK] EDA terminado. Revisá la carpeta:", out_dir)


if __name__ == "__main__":
    main()
