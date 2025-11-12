
# Carga a SQLite (TP Properati)

Este paso crea una base SQLite con las tablas requeridas por el TP.

## Uso
```bash
python load_to_sqlite.py --csv properati_clean.csv --artifacts_dir artifacts --db_path artifacts/database.db --store both --convert_currency --ars_usd 900 --uyu_usd 39
```

- `--store both` guarda: 
  - `input_data`: columnas originales que entran al preprocesador + `price`
  - `preprocessed_data`: matriz transformada en formato (row_idx, feature, value)
- `--convert_currency` aplica la conversión ARS/UYU -> USD en `input_data`.
- Los artefactos (`preprocessor.joblib`, `feature_names.json`, `X_preprocessed.*`) se leen desde `--artifacts_dir`.

## Esquema de tablas
- `input_data(row_idx, <features crudas...>, price)`
- `preprocessed_data(row_idx, feature, value)`
- `model_results(id, run_id, row_idx, y_true, y_pred, split, model_name, created_at)`
- `model_config(id, run_id, created_at, preprocessor_path, feature_names_path, ars_usd, uyu_usd, notes)`
