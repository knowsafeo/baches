# -*- coding: utf-8 -*-
"""
P007_Analisis.py — Análisis estadístico de baches y deterioros
──────────────────────────────────────────────────────────────
Lee el archivo resultados/detecciones.csv (lo genera P006_Interfaz.py)
y calcula:
    - Cantidad de baches y deterioros detectados
    - Nivel de confianza del modelo (promedio, mínimo, máximo)
    - Frecuencia de detecciones por sector
    - Sectores con mayor incidencia
    - Ranking de prioridad de mantenimiento vial

Genera 5 gráficos y los guarda en resultados/graficos

Requisitos:
    pip install pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ═══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE RUTAS
# ═══════════════════════════════════════════════════════════════
RUTA_BASE = r"D:\Proyectos senati 4to semestre\Capitulo 8 - Baches"
RUTA_CSV = os.path.join(RUTA_BASE, "resultados", "detecciones.csv")
RUTA_GRAFICOS = os.path.join(RUTA_BASE, "resultados", "graficos")
RUTA_RANKING = os.path.join(RUTA_BASE, "resultados", "prioridad_sectores.csv")

os.makedirs(RUTA_GRAFICOS, exist_ok=True)

# Pesos para calcular la prioridad
# Un bache grande es más peligroso que un deterioro pequeño
PESO_SEVERIDAD = {"PEQUENO": 1, "MEDIO": 2, "GRANDE": 3}
PESO_TIPO = {"bache": 1.5, "deterioro": 1.0}

# ═══════════════════════════════════════════════════════════════
#  1. CARGAR LOS DATOS
# ═══════════════════════════════════════════════════════════════
if not os.path.exists(RUTA_CSV):
    print("No existe el archivo detecciones.csv")
    print("Primero procesa videos con P006_Interfaz.py")
    exit()

df = pd.read_csv(RUTA_CSV, encoding="utf-8-sig")

print("=" * 55)
print("   ANÁLISIS DE BACHES Y DETERIOROS - TACNA")
print("=" * 55)
print(f"Total de registros: {len(df)}")
print("\nPrimeras filas:")
print(df.head())

# ═══════════════════════════════════════════════════════════════
#  2. MÉTRICAS GENERALES
# ═══════════════════════════════════════════════════════════════
total_baches = len(df[df["tipo"] == "bache"])
total_deterioros = len(df[df["tipo"] == "deterioro"])

# Confianza con NumPy
confianzas = df["confianza"].to_numpy()
conf_prom = np.mean(confianzas)
conf_min = np.min(confianzas)
conf_max = np.max(confianzas)
conf_desv = np.std(confianzas)

print("\n--- CANTIDAD DETECTADA ---")
print(f"Baches:     {total_baches}")
print(f"Deterioros: {total_deterioros}")

print("\n--- NIVEL DE CONFIANZA ---")
print(f"Promedio:   {conf_prom:.1%}")
print(f"Mínimo:     {conf_min:.1%}")
print(f"Máximo:     {conf_max:.1%}")
print(f"Desv. est.: {conf_desv:.3f}")

# Tabla tipo x severidad
tabla = pd.crosstab(df["tipo"], df["severidad"])
tabla = tabla.reindex(columns=["PEQUENO", "MEDIO", "GRANDE"], fill_value=0)
print("\n--- TIPO vs SEVERIDAD ---")
print(tabla)

# Frecuencia por sector
por_sector = df["sector"].value_counts()
print("\n--- FRECUENCIA POR SECTOR ---")
print(por_sector)

# ═══════════════════════════════════════════════════════════════
#  3. PRIORIDAD DE MANTENIMIENTO
# ═══════════════════════════════════════════════════════════════
# Cada detección suma puntos según su severidad y tipo
df["puntaje"] = df["severidad"].map(PESO_SEVERIDAD) * df["tipo"].map(PESO_TIPO)

ranking = df.groupby("sector").agg(
    baches=("tipo", lambda x: (x == "bache").sum()),
    deterioros=("tipo", lambda x: (x == "deterioro").sum()),
    grandes=("severidad", lambda x: (x == "GRANDE").sum()),
    confianza_prom=("confianza", "mean"),
    puntaje=("puntaje", "sum"),
)
ranking = ranking.sort_values("puntaje", ascending=False)

# Clasificar en ALTA / MEDIA / BAJA usando percentiles
p66 = np.percentile(ranking["puntaje"], 66)
p33 = np.percentile(ranking["puntaje"], 33)


def nivel_prioridad(puntos):
    if puntos >= p66:
        return "ALTA"
    elif puntos >= p33:
        return "MEDIA"
    return "BAJA"


ranking["prioridad"] = ranking["puntaje"].apply(nivel_prioridad)
ranking["confianza_prom"] = ranking["confianza_prom"].round(3)

print("\n--- PRIORIDAD DE MANTENIMIENTO POR SECTOR ---")
print(ranking)
ranking.to_csv(RUTA_RANKING, encoding="utf-8-sig")
print(f"\nRanking guardado en: {RUTA_RANKING}")

# ═══════════════════════════════════════════════════════════════
#  4. GRÁFICOS
# ═══════════════════════════════════════════════════════════════
COLOR_SEV = ["#6495ED", "#F5BE32", "#DC5050"]   # pequeño, medio, grande

# Gráfico 1: cantidad por tipo y severidad (barras agrupadas)
tabla.plot(kind="bar", color=COLOR_SEV, figsize=(8, 5))
plt.title("Cantidad de detecciones por tipo y severidad")
plt.xlabel("Tipo")
plt.ylabel("Cantidad")
plt.xticks(rotation=0)
plt.legend(title="Severidad")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "1_tipo_severidad.png"))

# Gráfico 2: histograma del nivel de confianza
plt.figure(figsize=(8, 5))
plt.hist(confianzas, bins=10, color="#9B8EC4", edgecolor="black")
plt.axvline(conf_prom, color="red", linestyle="--",
            label=f"Promedio: {conf_prom:.0%}")
plt.title("Distribución del nivel de confianza del modelo")
plt.xlabel("Confianza")
plt.ylabel("Cantidad de detecciones")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "2_confianza.png"))

# Gráfico 3: frecuencia por sector (barras horizontales)
plt.figure(figsize=(8, 5))
por_sector.sort_values().plot(kind="barh", color="#7B9ED9")
plt.title("Sectores con mayor incidencia de baches y deterioros")
plt.xlabel("Cantidad de detecciones")
plt.ylabel("Sector")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "3_sectores.png"))

# Gráfico 4: porcentaje por severidad (torta)
sev = df["severidad"].value_counts().reindex(
    ["PEQUENO", "MEDIO", "GRANDE"], fill_value=0)
plt.figure(figsize=(6, 6))
plt.pie(sev, labels=sev.index, autopct="%1.1f%%", colors=COLOR_SEV)
plt.title("Porcentaje de detecciones por severidad")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "4_severidad.png"))

# Gráfico 5: puntaje de prioridad por sector
colores_prio = {"ALTA": "#DC5050", "MEDIA": "#F5BE32", "BAJA": "#7DD9A8"}
colores = [colores_prio[p] for p in ranking["prioridad"]]
plt.figure(figsize=(9, 5))
plt.bar(ranking.index, ranking["puntaje"], color=colores)
plt.title("Prioridad de mantenimiento vial por sector")
plt.xlabel("Sector")
plt.ylabel("Puntaje de prioridad")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "5_prioridad.png"))

print(f"\nGráficos guardados en: {RUTA_GRAFICOS}")

# ═══════════════════════════════════════════════════════════════
#  5. PROPUESTA
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("   PROPUESTA DE MANTENIMIENTO VIAL")
print("=" * 55)
for i, (sector, fila) in enumerate(ranking.iterrows(), start=1):
    print(f"{i}. {sector} -> prioridad {fila['prioridad']} "
          f"({fila['baches']} baches, {fila['deterioros']} deterioros, "
          f"{fila['grandes']} grandes)")

plt.show()
