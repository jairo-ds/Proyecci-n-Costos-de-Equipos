# Dashboard - Proyección de Costos de Equipos

Dashboard interactivo (Streamlit) con el análisis exploratorio, el diagnóstico
de los modelos de regresión, y la proyección Monte Carlo para Equipo 1 y
Equipo 2, tal como se desarrolló en el notebook del caso.

## Cómo correrlo en tu VS Code local

1. **Abre esta carpeta en VS Code** (`File → Open Folder...`).

2. **Crea un entorno virtual** (recomendado, para no ensuciar tu Python global):

   ```bash
   python -m venv venv
   ```

   Actívalo:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. **Instala las dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Corre la app:**

   ```bash
   streamlit run app.py
   ```

5. Se abrirá automáticamente en tu navegador en `http://localhost:8501`
   (si no se abre solo, entra a esa dirección manualmente).

## Datos

La carpeta `data/` ya incluye los 4 archivos que usaste en el notebook
(`historico_equipos.csv`, `X.csv`, `Y.csv`, `Z.csv`), así que **funciona sin
configuración adicional**. Si quieres probar con otros datos, reemplaza esos
archivos (deben mantener el mismo nombre y formato), o usa el cargador de
archivos que aparece automáticamente en la barra lateral si los CSV no se
encuentran.

## Estructura del proyecto

```
dashboard/
├── app.py              <- Interfaz (Streamlit): pestañas, filtros, gráficos
├── data_models.py       <- Lógica: limpieza, regresión, diagnóstico, Monte Carlo
├── requirements.txt      <- Dependencias
├── data/                <- Los 4 CSV del caso
└── README.md            <- Este archivo
```

## Qué incluye el dashboard

- **🔍 Análisis Exploratorio:** series históricas de X, Y, Z, Equipo1 y
  Equipo2 (filtrable por variable y rango de fechas), matriz de correlación
  en niveles y en variaciones diarias, estadísticas descriptivas, y un
  resumen de qué tan lejos llegan los datos "futuros reales" de cada insumo.
- **⚙️ Modelo Equipo 1 / Equipo 2:** ecuación de la regresión, R², VIF,
  Durbin-Watson, prueba ADF sobre residuos (cointegración), gráfico de
  proyección con intervalo de confianza ajustable, y backtesting
  walk-forward bajo demanda (botón, para no recalcularlo en cada
  interacción).
- **📈 Comparación:** métricas de ambos modelos lado a lado, y sus
  proyecciones normalizadas (día 1 = 100) para comparar la forma relativa
  de la incertidumbre entre los dos equipos.

## Filtros disponibles (barra lateral)

- Rango de fechas del histórico mostrado en el EDA.
- Nivel de confianza del intervalo (80%-99%).
- Número de simulaciones Monte Carlo (500 a 10,000 — más simulaciones =
  más preciso pero más lento).
- Usar o no drift (tendencia histórica) en la simulación.
- Ancho de banda máximo aceptable para elegir el horizonte automáticamente.
- Opción de fijar el horizonte manualmente en vez de calcularlo.

## Notas técnicas

- La app usa `st.cache_data` para no releer los CSV en cada interacción —
  si reemplazas los archivos de `data/`, puede que necesites reiniciar la
  app (`r` en la terminal donde corre Streamlit, o Ctrl+C y volver a
  correrlo) para que tome los nuevos datos.
- El backtesting no se ejecuta automáticamente (puede tardar varios
  segundos) — se dispara con un botón dentro de cada pestaña de equipo.
