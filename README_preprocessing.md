
# Cómo usar el preprocesamiento

1) Colocar `properati_clean.csv` en el mismo directorio desde donde ejecutarás el script.
2) Ejecutar:

   ```bash
   python preprocessing.py --csv properati_clean.csv --ars_usd 90 --uyu_usd 50
   ```

3) Archivos de salida:
   - `artifacts/preprocessor.joblib` -> el ColumnTransformer ajustado
   - `artifacts/feature_names.json` -> nombres de columnas transformadas
   - `artifacts/X_preprocessed.npy` y `artifacts/y.npy` -> matrices listas para entrenar
     (si `y` contiene nulos, también se crea `X_preprocessed_aligned.npy` con las mismas filas que `y.npy`)

> Nota: el módulo prioriza las columnas recomendadas por dominio (rooms, bedrooms, bathrooms, surface_total,
  surface_covered, lat, lon, l2, l3, property_type). Si alguna falta en tu CSV, completa automáticamente
  con columnas numéricas/categóricas detectadas por dtype (excluye id/fechas/textos largos).
