"""
Agente de IA - Backend
========================

Diferencia clave con un "sistema de IA convencional":
  - Un sistema convencional recibe una pregunta y genera una respuesta a partir
    de lo que ya sabe (o de un contexto fijo) - no decide nada por sí mismo.
  - Este AGENTE tiene HERRAMIENTAS que decide usar por sí mismo según la
    pregunta (autonomía), mantiene MEMORIA de la conversación, y puede
    encadenar varias acciones (consultar datos propios + buscar en la web)
    antes de responder.

Herramientas disponibles:
  1. consultar_proyeccion: consulta los resultados YA calculados en el
     dashboard (coeficientes, diagnósticos, proyecciones) - así el agente
     nunca "inventa" cifras, siempre cita el análisis real.
  2. web_search: herramienta NATIVA de Anthropic (ejecutada por Anthropic,
     no por nuestro código) para noticias y contexto de mercado actual.
"""

import json
import anthropic


MODELO = "claude-sonnet-5"  # revisa https://docs.claude.com para el string exacto disponible en tu cuenta

HERRAMIENTA_PROYECCION = {
    "name": "consultar_proyeccion",
    "description": (
        "Consulta los resultados reales del análisis de proyección de costos "
        "(coeficientes de la regresión, diagnósticos estadísticos, y la tabla "
        "de proyección con intervalos de confianza) para Equipo 1 o Equipo 2. "
        "Usar esta herramienta SIEMPRE que la pregunta se refiera a cifras, "
        "modelos, horizontes, o resultados del análisis - nunca inventar estos "
        "números de memoria."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "equipo": {
                "type": "string",
                "enum": ["Equipo1", "Equipo2"],
                "description": "Qué equipo consultar",
            },
            "dia_horizonte": {
                "type": "integer",
                "description": (
                    "Día específico del horizonte a consultar (ej. 30 para el "
                    "pronóstico a 30 días hábiles). Si se omite, se devuelve un "
                    "resumen (día 1, mitad del horizonte, y último día)."
                ),
            },
        },
        "required": ["equipo"],
    },
}

HERRAMIENTA_WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search", "max_uses": 4}


def construir_contexto(reg1, reg2, proy1, proy2, horiz1, horiz2) -> dict:
    """Empaqueta los resultados del dashboard para que la herramienta los pueda
    consultar. Se llama una vez al cargar los datos, no en cada pregunta."""
    return {
        "Equipo1": {"reg": reg1, "proy": proy1, "horizonte": horiz1},
        "Equipo2": {"reg": reg2, "proy": proy2, "horizonte": horiz2},
    }


def ejecutar_consultar_proyeccion(contexto: dict, equipo: str, dia_horizonte: int | None = None) -> str:
    datos = contexto[equipo]
    reg, proy, horiz = datos["reg"], datos["proy"], datos["horizonte"]

    ecuacion = " + ".join(f"{v:.4f}*{k}" for k, v in reg.coeficientes.items() if k != "const")
    ecuacion = f"{reg.coeficientes['const']:.4f} + {ecuacion}"

    resumen = {
        "equipo": equipo,
        "ecuacion_regresion": ecuacion,
        "r2": round(reg.r2, 4),
        "durbin_watson": round(reg.durbin_watson, 3),
        "adf_residuos_pvalue": round(reg.adf_resid_pvalue, 5),
        "cointegrado": reg.cointegrado,
        "horizonte_recomendado_dias_habiles": horiz,
        "n_observaciones_historicas": reg.n_obs,
    }

    t = proy.tabla
    if dia_horizonte is not None:
        fila = t[t["h"] == dia_horizonte]
        if fila.empty:
            resumen["error"] = f"El horizonte proyectado solo llega a {t['h'].max()} días."
        else:
            r = fila.iloc[0]
            resumen["proyeccion_dia"] = {
                "h": int(r["h"]), "fecha": str(r["fecha"].date()),
                "forecast_mediana": round(r["forecast_mediana"], 2),
                "lower_95": round(r["lower"], 2), "upper_95": round(r["upper"], 2),
            }
    else:
        indices = sorted(set([0, len(t) // 2, len(t) - 1]))
        resumen["proyeccion_resumen"] = [
            {
                "h": int(t.iloc[i]["h"]), "fecha": str(t.iloc[i]["fecha"].date()),
                "forecast_mediana": round(t.iloc[i]["forecast_mediana"], 2),
                "lower_95": round(t.iloc[i]["lower"], 2), "upper_95": round(t.iloc[i]["upper"], 2),
            }
            for i in indices
        ]

    return json.dumps(resumen, ensure_ascii=False)


SYSTEM_PROMPT = """Eres un asistente analítico para un caso de consultoría sobre \
proyección de costos de equipos de construcción. Tienes dos herramientas:

1. consultar_proyeccion: para CUALQUIER cifra, coeficiente, diagnóstico u \
horizonte del análisis - SIEMPRE úsala en vez de inventar o recordar números.
2. web_search: para noticias, tendencias de mercado, o contexto económico \
actual que pueda complementar el pronóstico.

Cuando el usuario pregunte algo que combine ambas cosas (ej. "¿el pronóstico \
tiene sentido dado el contexto actual del mercado?"), usa AMBAS herramientas \
y compara/conecta la información en tu respuesta final. Sé conciso, concreto, \
y siempre distingue claramente qué viene de tu análisis (datos propios) y qué \
viene de la búsqueda web (fuente externa)."""


def correr_agente(client: anthropic.Anthropic, historial: list[dict], contexto_proyecciones: dict,
                   max_iteraciones: int = 5) -> tuple[str, list[dict], list[dict]]:
    """
    Corre el bucle agéntico: llama al modelo, y si pide usar la herramienta
    'consultar_proyeccion' (client-side), la ejecutamos nosotros y le devolvemos
    el resultado; repite hasta que el modelo entregue una respuesta final de texto.
    (web_search es server-side: Anthropic ya la resuelve dentro de la misma
    respuesta, no necesitamos ejecutarla nosotros.)

    Devuelve: (texto_respuesta_final, historial_actualizado, log_de_herramientas_usadas)
    """
    mensajes = list(historial)
    log_herramientas = []

    for _ in range(max_iteraciones):
        respuesta = client.messages.create(
            model=MODELO,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=[HERRAMIENTA_PROYECCION, HERRAMIENTA_WEB_SEARCH],
            messages=mensajes,
        )

        bloques_tool_use_cliente = [
            b for b in respuesta.content if b.type == "tool_use" and b.name == "consultar_proyeccion"
        ]
        hubo_web_search = any(b.type == "server_tool_use" and b.name == "web_search" for b in respuesta.content)
        if hubo_web_search:
            log_herramientas.append({"herramienta": "web_search"})

        mensajes.append({"role": "assistant", "content": respuesta.content})

        if not bloques_tool_use_cliente:
            texto_final = "".join(b.text for b in respuesta.content if b.type == "text")
            return texto_final, mensajes, log_herramientas

        resultados_tool = []
        for bloque in bloques_tool_use_cliente:
            log_herramientas.append({"herramienta": "consultar_proyeccion", "input": bloque.input})
            resultado = ejecutar_consultar_proyeccion(
                contexto_proyecciones,
                equipo=bloque.input.get("equipo"),
                dia_horizonte=bloque.input.get("dia_horizonte"),
            )
            resultados_tool.append({
                "type": "tool_result",
                "tool_use_id": bloque.id,
                "content": resultado,
            })
        mensajes.append({"role": "user", "content": resultados_tool})

    return "No se pudo completar la respuesta (demasiadas iteraciones de herramientas).", mensajes, log_herramientas
