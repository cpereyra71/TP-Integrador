# Proyecto TPPA
# EDA desde SQLite — Guía paso a paso

Este documento explica **cómo ejecutar el Análisis Exploratorio de Datos (EDA)** usando el script `eda_from_sqlite.py` que lee la tabla `input_data` de `artifacts/database.db` y **genera gráficos y resúmenes** en `artifacts/eda/`.

---

## 1) Requisitos

- **Python 3.x** (en tu entorno virtual)
- Librerías:
  - `pandas`
  - `numpy`
  - `matplotlib`

Instalación rápida (con el entorno activado):

```powershell
pip install pandas numpy matplotlib
```

*(Opcional si vas a entrenar después: `scikit-learn`, `joblib`, `scipy`, `sqlalchemy`).*

---

## 2) Ubicación de archivos

Estructura sugerida del proyecto:

```
TP_Properati/
 ├─ data/
 │   └─ properati_clean.csv
 ├─ artifacts/
 │   ├─ preprocessor.joblib
 │   ├─ feature_names.json
 │   ├─ X_preprocessed.npz
 │   ├─ y.npy
 │   └─ database.db        ← Base creada por load_to_sqlite.py
 ├─ Scripts/
 │   ├─ preprocessing.py
 │   ├─ load_to_sqlite.py
 │   └─ eda_from_sqlite.py ← Este script
 ├─ README.md
 └─ requirements.txt
```

> **Importante:** Asegurate que `artifacts/database.db` exista y tenga la tabla `input_data`.

---

## 3) Ejecución (Windows PowerShell)

Desde la **raíz del proyecto**:

```powershell
python .\Scripts\eda_from_sqlite.py --db_path .\data\artifacts\database.db --table input_data --out_dir .\data\artifacts\eda
```

Parámetros:
- `--db_path`: ruta al archivo SQLite.
- `--table`: tabla a leer (por defecto, `input_data`).
- `--out_dir`: carpeta donde se guardan resultados.

### Alternativa: variables de entorno

```powershell
$env:DB_PATH=".rtifacts\database.db"
$env:INPUT_TABLE="input_data"
$env:EDA_OUT=".rtifacts\eda"
python .\Scripts\eda_from_sqlite.py --db_path $env:DB_PATH --table $env:INPUT_TABLE --out_dir $env:EDA_OUT
```

---

## 4) ¿Qué resultados se generan?

En `artifacts/eda/` se guardan:
- **TXT/CSV**
  - `00_resumen_eda.txt` (resumen rápido)
  - `01_shape.txt`
  - `02_dtypes.csv`
  - `03_nulls.csv`
  - `04_describe_numeric.csv`
  - `05_top_categories.csv` (si hay columnas categóricas)
- **Gráficos (.png)**
  - `A1_hist_price.png` (histograma de precio)
  - `A2_hist_price_log.png` (histograma log10 de precio)
  - `B1_avg_price_by_property_type.png` (barras)
  - `B2_box_price_by_property_type.png` (boxplot)
  - `C1_scatter_surface_total_price.png` (dispersión)
  - `D1_corr_heatmap.png` (correlaciones numéricas)
  - `E1_geo_scatter_price.png` (lat/lon coloreado por precio; si existen columnas)

> El script verifica si las columnas necesarias existen y **omite** el gráfico que no se pueda construir, mostrando un aviso `[WARN]` en consola.

---

## 5) Interpretación sugerida (para el informe/presentación)

- **Distribución de precio:** revisar sesgo; usar `A2_hist_price_log.png` para ver mejor colas largas.
- **Precio por tipo de propiedad:** detectar segmentos (casas vs. departamentos vs. PH, etc.).
- **Superficie vs. precio:** observar la relación y la dispersión; decidir si conviene transformar o recortar outliers.
- **Correlaciones:** identificar variables numéricas que aportan más señal.
- **Mapa lat/lon:** verificar la consistencia geográfica y valores atípicos por ubicación.

---

## 6) Troubleshooting

- **`No se encontró la base de datos`**  
  Revisá la ruta de `--db_path` y que `database.db` exista.

- **`no module named ...`**  
  Instalá librerías faltantes: `pip install pandas numpy matplotlib`.

- **Gráfico no generado**  
  Puede faltar alguna columna (ej. `surface_total` o `lat/lon`). El script lo informa y continúa.

- **Resultados vacíos**  
  Verificá que la tabla `input_data` tenga filas:  
  ```sql
  SELECT COUNT(*) FROM input_data;
  ```
  (Podés chequearlo con DB Browser for SQLite o con `pandas.read_sql_query`.)

---

## 7) Checklist de entrega (EDA)

- [ ] Se adjuntan los CSV/TXT generados en `artifacts/eda/`
- [ ] Se incluyen gráficos clave en el informe / PPT
- [ ] Se explica brevemente cada hallazgo (2-3 bullets por gráfico)
- [ ] Se describen variables con mayor correlación con `price`
- [ ] Se listan posibles outliers y criterio (si corresponde)

---

Con esto, el EDA queda **reproducible** y **listo para presentar**. Cuando quieras, avanzamos al siguiente paso: **entrenar y comparar al menos dos algoritmos de regresión, guardando métricas en `model_results`**.
