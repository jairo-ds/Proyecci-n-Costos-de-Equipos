"""
Dashboard - Proyección de Costos de Equipos (Caso Dataknow)
=============================================================

Cómo correrlo localmente (ver también README.md):
    pip install -r requirements.txt
    streamlit run app.py

Se abre automáticamente en el navegador en http://localhost:8501
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from data_models import (
    cargar_historico, cargar_x, cargar_y, cargar_z, futuro_real,
    ajustar_regresion, parametros_random_walk, proyectar,
    horizonte_recomendado, backtest_regresion,
)

st.set_page_config(page_title="Proyección de Costos - Equipos", layout="wide", page_icon="📊")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ----------------------------------------------------------------------------
# Carga de datos (con caché para no releer los CSV en cada interacción)
# ----------------------------------------------------------------------------

@st.cache_data
def cargar_todo(path_historico, path_x, path_y, path_z):
    historico = cargar_historico(path_historico)
    x = cargar_x(path_x)
    y = cargar_y(path_y)
    z = cargar_z(path_z)
    return historico, x, y, z


def resolver_rutas():
    """Busca los CSV en ./data primero, y si no están, en la raíz del proyecto
    (por si se organizaron sin subcarpeta, como en este repo)."""
    base_dir = os.path.dirname(__file__)
    nombres = {
        "historico": "historico_equipos.csv",
        "x": "X.csv",
        "y": "Y.csv",
        "z": "Z.csv",
    }

    def existe_en(carpeta):
        return {k: os.path.join(carpeta, v) for k, v in nombres.items()}

    candidatos = [existe_en(DATA_DIR), existe_en(base_dir)]
    for rutas_candidatas in candidatos:
        if all(os.path.exists(p) for p in rutas_candidatas.values()):
            return rutas_candidatas

    st.sidebar.warning("No se encontraron los CSV en ./data ni en la raíz del proyecto — súbelos manualmente:")
    subidos = {}
    subidos["historico"] = st.sidebar.file_uploader("historico_equipos.csv", type="csv", key="up_hist")
    subidos["x"] = st.sidebar.file_uploader("X.csv", type="csv", key="up_x")
    subidos["y"] = st.sidebar.file_uploader("Y.csv", type="csv", key="up_y")
    subidos["z"] = st.sidebar.file_uploader("Z.csv", type="csv", key="up_z")
    if not all(subidos.values()):
        st.info("Sube los 4 archivos en la barra lateral para continuar, o cópialos en la carpeta ./data del proyecto.")
        st.stop()
    return subidos


rutas = resolver_rutas()
historico, x_df, y_df, z_df = cargar_todo(rutas["historico"], rutas["x"], rutas["y"], rutas["z"])
ultima_fecha = historico["Date"].iloc[-1]

st.title("📊 Proyección de Costos de Equipos")
st.caption("Caso de negocio: gestión de costos operativos en un proyecto de construcción")


# ----------------------------------------------------------------------------
# Sidebar: filtros globales
# ----------------------------------------------------------------------------

st.sidebar.header("Filtros")

rango_fechas = st.sidebar.date_input(
    "Rango de fechas (histórico)",
    value=(historico["Date"].min().date(), historico["Date"].max().date()),
    min_value=historico["Date"].min().date(),
    max_value=historico["Date"].max().date(),
)
if len(rango_fechas) == 2:
    f_ini, f_fin = pd.Timestamp(rango_fechas[0]), pd.Timestamp(rango_fechas[1])
else:
    f_ini, f_fin = historico["Date"].min(), historico["Date"].max()

historico_filtrado = historico[(historico["Date"] >= f_ini) & (historico["Date"] <= f_fin)]

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros de proyección")
nivel_confianza = st.sidebar.slider("Nivel de confianza", 0.80, 0.99, 0.95, 0.01)
n_sims = st.sidebar.select_slider("N° de simulaciones Monte Carlo", options=[500, 1000, 2000, 5000, 10000], value=5000)
usar_drift = st.sidebar.checkbox("Usar drift (tendencia histórica) en la simulación", value=False)
ancho_max = st.sidebar.slider("Ancho de banda máx. aceptable para elegir horizonte (%)", 5, 50, 20, 5) / 100
horizonte_manual = st.sidebar.checkbox("Fijar horizonte manualmente (en vez de calcularlo)", value=False)
h_manual = st.sidebar.slider("Horizonte manual (días hábiles)", 5, 200, 89, 1) if horizonte_manual else None


# ----------------------------------------------------------------------------
# Preparar params e insumos para cada equipo (reutilizado en varias pestañas)
# ----------------------------------------------------------------------------

params_y = parametros_random_walk(historico["Price_Y"])
params_z = parametros_random_walk(historico["Price_Z"])
params_x = parametros_random_walk(historico["Price_X"])
futuro_y_arr = futuro_real(y_df, ultima_fecha)
futuro_z_arr = futuro_real(z_df, ultima_fecha)
futuro_x_arr = futuro_real(x_df, ultima_fecha)

reg1 = ajustar_regresion(historico, "Price_Equipo1", ["Price_Y", "Price_X"])
reg2 = ajustar_regresion(historico, "Price_Equipo2", ["Price_Z", "Price_Y", "Price_X"])

insumos_e1 = {
    "Price_Y": {"params": params_y, "reales_futuros": futuro_y_arr},
    "Price_X": {"params": params_x, "reales_futuros": futuro_x_arr},
}
insumos_e2 = {
    "Price_Z": {"params": params_z, "reales_futuros": futuro_z_arr},
    "Price_Y": {"params": params_y, "reales_futuros": futuro_y_arr},
    "Price_X": {"params": params_x, "reales_futuros": futuro_x_arr},
}


def obtener_horizonte(reg, coefs, insumos):
    if horizonte_manual:
        return h_manual
    return horizonte_recomendado(reg, coefs, insumos, ultima_fecha, horizonte_max=150,
                                  n_sims=2000, ancho_relativo_max=ancho_max,
                                  nivel_confianza=nivel_confianza)


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------

tab_eda, tab_e1, tab_e2, tab_comp, tab_agente = st.tabs([
    "🔍 Análisis Exploratorio", "⚙️ Modelo Equipo 1", "⚙️ Modelo Equipo 2",
    "📈 Comparación", "🤖 Agente de IA"
])


# ============================================================================
# TAB 1: Análisis Exploratorio
# ============================================================================
with tab_eda:
    st.subheader("Series históricas")

    col1, col2 = st.columns(2)
    with col1:
        variables_equipo = st.multiselect(
            "Precios de equipos", ["Price_Equipo1", "Price_Equipo2"],
            default=["Price_Equipo1", "Price_Equipo2"]
        )
    with col2:
        variables_insumo = st.multiselect(
            "Precios de materias primas", ["Price_X", "Price_Y", "Price_Z"],
            default=["Price_X", "Price_Y", "Price_Z"]
        )

    fig = go.Figure()
    for col in variables_equipo + variables_insumo:
        fig.add_trace(go.Scatter(x=historico_filtrado["Date"], y=historico_filtrado[col],
                                  mode="lines", name=col))
    fig.update_layout(height=450, xaxis_title="Fecha", yaxis_title="Precio",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Matriz de correlación (niveles)")
        cols_num = ["Price_X", "Price_Y", "Price_Z", "Price_Equipo1", "Price_Equipo2"]
        corr = historico_filtrado[cols_num].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, aspect="auto")
        fig_corr.update_layout(height=420)
        st.plotly_chart(fig_corr, width='stretch')

    with c2:
        st.subheader("Correlación de variaciones diarias")
        st.caption("Más confiable que niveles: evita el sesgo de tendencia común entre series no estacionarias.")
        diffs = historico_filtrado[cols_num].diff().dropna()
        corr_diff = diffs.corr()
        fig_corr_diff = px.imshow(corr_diff, text_auto=".2f", color_continuous_scale="RdBu_r",
                                   zmin=-1, zmax=1, aspect="auto")
        fig_corr_diff.update_layout(height=420)
        st.plotly_chart(fig_corr_diff, width='stretch')

    st.markdown("---")
    st.subheader("Estadísticas descriptivas")
    st.dataframe(historico_filtrado[cols_num].describe().T, width='stretch')

    with st.expander("Ver información de disponibilidad de datos futuros (X, Y, Z)"):
        st.write(f"**Último día del histórico de equipos:** {ultima_fecha.date()}")
        st.write(f"- Días reales futuros disponibles en **X.csv**: {len(futuro_real(x_df, ultima_fecha))} "
                 f"(hasta {x_df['Date'].max().date()})")
        st.write(f"- Días reales futuros disponibles en **Y.csv**: {len(futuro_y_arr)} "
                 f"(hasta {y_df['Date'].max().date()})")
        st.write(f"- Días reales futuros disponibles en **Z.csv**: {len(futuro_z_arr)} "
                 f"(hasta {z_df['Date'].max().date()})")
        if len(futuro_z_arr) == 0:
            st.info("Z.csv no aporta datos posteriores al histórico — se simula desde el primer día proyectado.")


# ============================================================================
# Función reutilizable para renderizar la pestaña de un equipo
# ============================================================================

def render_tab_equipo(nombre_equipo: str, col_equipo: str, reg, coefs: dict, insumos: dict, color: str):
    st.subheader(f"Regresión lineal — {nombre_equipo}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{reg.r2:.3f}")
    c2.metric("Durbin-Watson", f"{reg.durbin_watson:.3f}")
    c3.metric("ADF residuos (p-value)", f"{reg.adf_resid_pvalue:.4f}")
    c4.metric("¿Cointegrado?", "✅ Sí" if reg.cointegrado else "⚠️ No")

    st.markdown("**Ecuación ajustada:**")
    partes = [f"{reg.coeficientes['const']:.3f}"]
    for k, v in reg.coeficientes.items():
        if k != "const":
            signo = "+" if v >= 0 else "-"
            partes.append(f" {signo} {abs(v):.4f}·{k}")
    st.code(f"{col_equipo} = " + "".join(partes))

    with st.expander("Ver tabla de coeficientes, p-values y VIF"):
        tabla_coefs = pd.DataFrame({
            "coeficiente": reg.coeficientes,
            "p-value": reg.pvalues,
            "VIF": reg.vif,
        })
        st.dataframe(tabla_coefs, width='stretch')
        st.caption("VIF < 5 indica ausencia de multicolinealidad grave. "
                   "ADF sobre residuos con p < 0.05 confirma cointegración (relación real, no espuria).")

    st.markdown("---")
    st.subheader("Proyección con intervalo de confianza")

    horiz = obtener_horizonte(reg, coefs, insumos)
    st.info(f"Horizonte {'fijado manualmente' if horizonte_manual else 'recomendado'}: "
            f"**{horiz} días hábiles** (~{horiz/21:.1f} meses), con banda ≤ {ancho_max*100:.0f}% del valor central.")

    proy = proyectar(reg, coefs, insumos, ultima_fecha, horiz, n_sims, nivel_confianza, usar_drift)

    hist_reciente = historico.tail(180)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_reciente["Date"], y=hist_reciente[col_equipo],
                              mode="lines", name="Histórico", line=dict(color="#2b6cb0")))
    fig.add_trace(go.Scatter(x=proy.tabla["fecha"], y=proy.tabla["forecast_mediana"],
                              mode="lines", name="Proyección (mediana)",
                              line=dict(color=color, dash="dash")))
    fig.add_trace(go.Scatter(x=proy.tabla["fecha"], y=proy.tabla["upper"],
                              mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=proy.tabla["fecha"], y=proy.tabla["lower"],
                              mode="lines", line=dict(width=0), fill="tonexty",
                              fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)",
                              name=f"IC {nivel_confianza*100:.0f}%", hoverinfo="skip"))
    fig.add_vline(x=ultima_fecha, line_dash="dot", line_color="gray",
                  annotation_text="Inicio proyección")

    for nombre_insumo, n_reales in proy.n_reales_por_insumo.items():
        if n_reales > 0 and n_reales < horiz:
            fecha_corte = proy.tabla["fecha"].iloc[n_reales - 1]
            fig.add_vline(x=fecha_corte, line_dash="dot", line_color="#805ad5",
                          annotation_text=f"Fin {nombre_insumo} real")

    fig.update_layout(height=480, xaxis_title="Fecha", yaxis_title="Precio",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')

    with st.expander("Ver tabla completa de la proyección"):
        st.dataframe(proy.tabla, width='stretch')
        st.download_button(f"Descargar proyección {nombre_equipo} (CSV)",
                            proy.tabla.to_csv(index=False), file_name=f"proyeccion_{nombre_equipo}.csv")

    st.markdown("---")
    st.subheader("Validación con backtesting walk-forward")
    st.caption("Compara el pronóstico contra lo que realmente pasó, en múltiples puntos del histórico.")
    if st.button(f"Ejecutar backtesting para {nombre_equipo}", key=f"bt_{nombre_equipo}"):
        with st.spinner("Corriendo backtesting..."):
            independientes = [k for k in coefs.keys() if k != "const"]
            resumen_bt = backtest_regresion(historico, col_equipo, independientes,
                                             horizonte=min(30, horiz), n_origenes=12, n_sims=500)
        if resumen_bt.empty:
            st.warning("No hay suficiente historia para este backtesting.")
        else:
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=resumen_bt["h"], y=resumen_bt["MAPE_%"], name="MAPE %"))
            fig_bt.add_trace(go.Scatter(x=resumen_bt["h"], y=resumen_bt["cobertura"] * 100,
                                         name="Cobertura IC %", yaxis="y2"))
            fig_bt.update_layout(
                height=350, xaxis_title="Horizonte (días)",
                yaxis=dict(title="MAPE %"),
                yaxis2=dict(title="Cobertura %", overlaying="y", side="right", range=[0, 100]),
                legend=dict(orientation="h", y=1.15),
            )
            st.plotly_chart(fig_bt, width='stretch')
            st.dataframe(resumen_bt, width='stretch')

    return proy, horiz


with tab_e1:
    proy1, horiz1 = render_tab_equipo("Equipo 1", "Price_Equipo1", reg1, reg1.coeficientes, insumos_e1, "#38a169")

with tab_e2:
    proy2, horiz2 = render_tab_equipo("Equipo 2", "Price_Equipo2", reg2, reg2.coeficientes, insumos_e2, "#dd6b20")


# ============================================================================
# TAB 4: Comparación
# ============================================================================
with tab_comp:
    st.subheader("Comparación de modelos")

    tabla_comp = pd.DataFrame({
        "Métrica": ["R²", "Durbin-Watson", "ADF residuos (p-value)", "Cointegrado",
                    "Horizonte recomendado (días hábiles)"],
        "Equipo 1": [f"{reg1.r2:.3f}", f"{reg1.durbin_watson:.3f}", f"{reg1.adf_resid_pvalue:.4f}",
                     "Sí" if reg1.cointegrado else "No", str(horiz1)],
        "Equipo 2": [f"{reg2.r2:.3f}", f"{reg2.durbin_watson:.3f}", f"{reg2.adf_resid_pvalue:.4f}",
                     "Sí" if reg2.cointegrado else "No", str(horiz2)],
    })
    st.dataframe(tabla_comp, width='stretch', hide_index=True)

    st.markdown("---")
    st.subheader("Proyecciones lado a lado (normalizadas al día 1 = 100)")
    st.caption("Útil para comparar la FORMA relativa de la incertidumbre entre equipos, "
               "independientemente de su escala de precio.")

    norm1 = proy1.tabla.copy()
    norm1["forecast_norm"] = norm1["forecast_mediana"] / norm1["forecast_mediana"].iloc[0] * 100
    norm1["upper_norm"] = norm1["upper"] / norm1["forecast_mediana"].iloc[0] * 100
    norm1["lower_norm"] = norm1["lower"] / norm1["forecast_mediana"].iloc[0] * 100

    norm2 = proy2.tabla.copy()
    norm2["forecast_norm"] = norm2["forecast_mediana"] / norm2["forecast_mediana"].iloc[0] * 100
    norm2["upper_norm"] = norm2["upper"] / norm2["forecast_mediana"].iloc[0] * 100
    norm2["lower_norm"] = norm2["lower"] / norm2["forecast_mediana"].iloc[0] * 100

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=norm1["h"], y=norm1["upper_norm"], line=dict(width=0),
                                   showlegend=False, hoverinfo="skip"))
    fig_comp.add_trace(go.Scatter(x=norm1["h"], y=norm1["lower_norm"], line=dict(width=0),
                                   fill="tonexty", fillcolor="rgba(56,161,105,0.15)",
                                   name="IC Equipo 1", hoverinfo="skip"))
    fig_comp.add_trace(go.Scatter(x=norm1["h"], y=norm1["forecast_norm"], name="Equipo 1",
                                   line=dict(color="#38a169")))

    fig_comp.add_trace(go.Scatter(x=norm2["h"], y=norm2["upper_norm"], line=dict(width=0),
                                   showlegend=False, hoverinfo="skip"))
    fig_comp.add_trace(go.Scatter(x=norm2["h"], y=norm2["lower_norm"], line=dict(width=0),
                                   fill="tonexty", fillcolor="rgba(221,107,32,0.15)",
                                   name="IC Equipo 2", hoverinfo="skip"))
    fig_comp.add_trace(go.Scatter(x=norm2["h"], y=norm2["forecast_norm"], name="Equipo 2",
                                   line=dict(color="#dd6b20")))

    fig_comp.update_layout(height=450, xaxis_title="Días hábiles proyectados",
                           yaxis_title="Índice (día 1 = 100)", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_comp, width='stretch')

    st.caption(
        "Nota: el horizonte 'recomendado' de cada equipo se calcula de forma independiente "
        "según el criterio de ancho de banda configurado en la barra lateral, así que pueden "
        "tener distinta longitud."
    )


# ============================================================================
# TAB 5: Agente de IA
# ============================================================================
with tab_agente:
    st.subheader("🤖 Agente de IA — Pregunta sobre los resultados")

    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key:
        api_key = st.text_input(
            "Tu API key de Anthropic (no se guarda ni se sube a ningún lado, solo vive en esta sesión)",
            type="password",
        )
        st.caption(
            "⚠️ Nunca pongas tu API key directamente en el código si vas a subirlo a GitHub. "
            "Para producción, usa `.streamlit/secrets.toml` (local) o la sección 'Secrets' de "
            "Streamlit Cloud (para el despliegue) — ambos están fuera del control de versiones."
        )

    if not api_key:
        st.info("Ingresa tu API key de Anthropic arriba para activar el agente.")
    else:
        import anthropic
        from agent import construir_contexto, correr_agente

        contexto_proyecciones = construir_contexto(reg1, reg2, proy1, proy2, horiz1, horiz2)
        client = anthropic.Anthropic(api_key=api_key)

        if "chat_historial" not in st.session_state:
            st.session_state.chat_historial = []

        for msg in st.session_state.chat_historial:
            if msg["role"] == "user" and isinstance(msg["content"], str):
                with st.chat_message("user"):
                    st.write(msg["content"])
            elif msg["role"] == "assistant":
                texto = "".join(b.text for b in msg["content"] if getattr(b, "type", None) == "text")
                if texto:
                    with st.chat_message("assistant"):
                        st.write(texto)

        pregunta = st.chat_input("Pregunta sobre los resultados...")
        if pregunta:
            st.session_state.chat_historial.append({"role": "user", "content": pregunta})
            with st.chat_message("user"):
                st.write(pregunta)

            with st.chat_message("assistant"):
                with st.spinner("Pensando (puede usar herramientas)..."):
                    try:
                        texto_final, historial_actualizado, log_tools = correr_agente(
                            client, st.session_state.chat_historial, contexto_proyecciones
                        )
                        st.session_state.chat_historial = historial_actualizado
                        st.write(texto_final)
                        if log_tools:
                            with st.expander("🔧 Herramientas usadas en esta respuesta"):
                                for t in log_tools:
                                    st.write(t)
                    except Exception as e:
                        st.error(f"Error llamando a la API de Anthropic: {e}")

        if st.button("🗑️ Reiniciar conversación"):
            st.session_state.chat_historial = []
            st.rerun()

        st.markdown("---")
        st.caption("Preguntas de ejemplo para probar el agente:")
        ejemplos = [
            "¿Cuál es el horizonte recomendado para Equipo1 y por qué?",
            "¿Qué tan confiable es el modelo de Equipo2? (R², cointegración)",
            "Busca noticias recientes sobre precios de materias primas en construcción y dime si son consistentes con el pronóstico de Equipo1",
            "¿Cuál es el precio proyectado de Equipo2 en el día 30 del horizonte?",
        ]
        for e in ejemplos:
            st.code(e, language=None)
