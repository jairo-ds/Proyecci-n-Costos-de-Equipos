"""
Lógica de datos y modelos para el dashboard - Proyección de Costos de Equipos.

Este módulo replica exactamente la metodología validada en el notebook:
  - Regresión lineal Equipo_i ~ materias primas
  - Diagnóstico (VIF, Durbin-Watson, ADF sobre residuos -> cointegración)
  - Simulación Monte Carlo para proyectar hacia el futuro, usando datos
    reales futuros de Y donde existan y simulando el resto como random walk.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson


# ----------------------------------------------------------------------------
# Carga y limpieza
# ----------------------------------------------------------------------------

def cargar_historico(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def cargar_x(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def cargar_y(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df["Price"] = df["Price"].astype(str).str.replace(",", ".").astype(float)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df.sort_values("Date").reset_index(drop=True)


def cargar_z(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def futuro_real(df_insumo: pd.DataFrame, ultima_fecha: pd.Timestamp) -> np.ndarray:
    """Valores reales de un insumo posteriores al fin del histórico (si existen)."""
    fut = df_insumo[df_insumo["Date"] > ultima_fecha].reset_index(drop=True)
    return fut["Price"].values


# ----------------------------------------------------------------------------
# Regresión y diagnóstico
# ----------------------------------------------------------------------------

@dataclass
class ResultadoRegresion:
    modelo: object
    r2: float
    coeficientes: dict
    pvalues: dict
    vif: dict
    durbin_watson: float
    adf_resid_pvalue: float
    n_obs: int

    @property
    def cointegrado(self) -> bool:
        return self.adf_resid_pvalue < 0.05


def ajustar_regresion(df: pd.DataFrame, dependiente: str, independientes: list[str]) -> ResultadoRegresion:
    y = df[dependiente]
    X = sm.add_constant(df[independientes])
    modelo = sm.OLS(y, X).fit()

    vif = {col: variance_inflation_factor(X.values, i) for i, col in enumerate(X.columns)}
    dw = durbin_watson(modelo.resid)
    adf_p = adfuller(modelo.resid)[1]

    return ResultadoRegresion(
        modelo=modelo,
        r2=modelo.rsquared,
        coeficientes=modelo.params.to_dict(),
        pvalues=modelo.pvalues.to_dict(),
        vif=vif,
        durbin_watson=dw,
        adf_resid_pvalue=adf_p,
        n_obs=int(modelo.nobs),
    )


def parametros_random_walk(serie: pd.Series) -> dict:
    diffs = serie.diff().dropna()
    return {
        "ultimo_valor": float(serie.iloc[-1]),
        "sigma": float(diffs.std(ddof=1)),
        "drift": float(diffs.mean()),
    }


# ----------------------------------------------------------------------------
# Simulación Monte Carlo
# ----------------------------------------------------------------------------

def simular_variable(ultimo_valor, sigma, drift, horizonte, n_sims, reales_futuros,
                      usar_drift, rng) -> np.ndarray:
    n_reales = min(len(reales_futuros), horizonte)
    n_simular = horizonte - n_reales

    paths = np.zeros((n_sims, horizonte))
    if n_reales > 0:
        paths[:, :n_reales] = reales_futuros[:n_reales]
        ancla = reales_futuros[n_reales - 1]
    else:
        ancla = ultimo_valor

    if n_simular > 0:
        drift_usado = drift if usar_drift else 0.0
        shocks = rng.normal(drift_usado, sigma, size=(n_sims, n_simular))
        paths[:, n_reales:] = ancla + np.cumsum(shocks, axis=1)

    return paths, n_reales


@dataclass
class ResultadoProyeccion:
    tabla: pd.DataFrame          # h, fecha, forecast_mediana, lower, upper
    n_reales_por_insumo: dict    # cuántos días reales se usaron por insumo
    horizonte: int
    nivel_confianza: float


def proyectar(reg: ResultadoRegresion, dependiente_coefs: dict, insumos: dict,
              ultima_fecha: pd.Timestamp, horizonte: int, n_sims: int,
              nivel_confianza: float, usar_drift: bool, seed: int = 42) -> ResultadoProyeccion:
    """
    insumos: { 'Price_Y': {'params': {...}, 'reales_futuros': np.array}, 'Price_Z': {...} }
    dependiente_coefs: modelo.params (incluye 'const' y una entrada por cada insumo)
    """
    rng = np.random.default_rng(seed)
    residuos_hist = reg.modelo.resid.values

    paths_por_insumo = {}
    n_reales_por_insumo = {}
    for nombre, info in insumos.items():
        paths, n_reales = simular_variable(
            ultimo_valor=info["params"]["ultimo_valor"],
            sigma=info["params"]["sigma"],
            drift=info["params"]["drift"],
            horizonte=horizonte,
            n_sims=n_sims,
            reales_futuros=info["reales_futuros"],
            usar_drift=usar_drift,
            rng=rng,
        )
        paths_por_insumo[nombre] = paths
        n_reales_por_insumo[nombre] = n_reales

    resid_boot = rng.choice(residuos_hist, size=(n_sims, horizonte), replace=True)

    pred = np.full((n_sims, horizonte), dependiente_coefs["const"])
    for nombre, paths in paths_por_insumo.items():
        pred += dependiente_coefs[nombre] * paths
    pred += resid_boot

    alpha = 1 - nivel_confianza
    q_low, q_high = alpha / 2 * 100, (1 - alpha / 2) * 100
    percentiles = np.percentile(pred, [q_low, 50, q_high], axis=0)

    tabla = pd.DataFrame({
        "h": np.arange(1, horizonte + 1),
        "forecast_mediana": percentiles[1],
        "lower": percentiles[0],
        "upper": percentiles[2],
    })
    tabla["fecha"] = pd.bdate_range(start=ultima_fecha + pd.Timedelta(days=1), periods=horizonte)

    return ResultadoProyeccion(
        tabla=tabla, n_reales_por_insumo=n_reales_por_insumo,
        horizonte=horizonte, nivel_confianza=nivel_confianza,
    )


def horizonte_recomendado(reg: ResultadoRegresion, dependiente_coefs: dict, insumos: dict,
                           ultima_fecha: pd.Timestamp, horizonte_max: int = 150,
                           n_sims: int = 3000, ancho_relativo_max: float = 0.20,
                           nivel_confianza: float = 0.95, seed: int = 42) -> int:
    """Mayor h tal que el semi-ancho de banda no supere ancho_relativo_max del valor central."""
    proy = proyectar(reg, dependiente_coefs, insumos, ultima_fecha, horizonte_max,
                      n_sims, nivel_confianza, usar_drift=False, seed=seed)
    t = proy.tabla
    ancho_rel = (t["upper"] - t["forecast_mediana"]) / t["forecast_mediana"]
    validos = t[ancho_rel <= ancho_relativo_max]
    return int(validos["h"].max()) if not validos.empty else 1


# ----------------------------------------------------------------------------
# Backtesting walk-forward (para la pestaña de métricas)
# ----------------------------------------------------------------------------

def backtest_regresion(df: pd.DataFrame, dependiente: str, independientes: list[str],
                        horizonte: int, n_origenes: int = 12, n_sims: int = 500,
                        seed: int = 42) -> pd.DataFrame:
    n = len(df)
    min_origen = max(500, n // 4)
    max_origen = n - horizonte
    if max_origen <= min_origen:
        return pd.DataFrame()
    origenes = sorted(set(np.linspace(min_origen, max_origen, n_origenes, dtype=int)))
    rng = np.random.default_rng(seed)

    registros = []
    for o in origenes:
        historia = df.iloc[:o + 1]
        futuro = df.iloc[o + 1: o + 1 + horizonte]
        if len(futuro) < horizonte:
            continue
        reg_o = ajustar_regresion(historia, dependiente, independientes)
        insumos_o = {}
        for col in independientes:
            p = parametros_random_walk(historia[col])
            insumos_o[col] = {"params": p, "reales_futuros": np.array([])}

        proy = proyectar(reg_o, reg_o.coeficientes, insumos_o, historia["Date"].iloc[-1],
                          horizonte, n_sims, 0.95, usar_drift=False, seed=seed)
        t = proy.tabla.copy()
        t["actual"] = futuro[dependiente].values
        t["error_abs"] = (t["actual"] - t["forecast_mediana"]).abs()
        t["dentro_banda"] = (t["actual"] >= t["lower"]) & (t["actual"] <= t["upper"])
        registros.append(t)

    if not registros:
        return pd.DataFrame()

    todo = pd.concat(registros, ignore_index=True)
    todo["ape"] = todo["error_abs"] / todo["actual"].abs()
    resumen = todo.groupby("h").agg(MAE=("error_abs", "mean"),
                                     cobertura=("dentro_banda", "mean")).reset_index()
    resumen["MAPE_%"] = todo.groupby("h")["ape"].mean().values * 100
    return resumen
