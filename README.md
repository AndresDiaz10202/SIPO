# SIPO — Sistema Inteligente de Predicción y Optimización de Movilidad Urbana

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![SUMO](https://img.shields.io/badge/SUMO-1.19-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

Sistema de inteligencia artificial para la predicción de congestión vial
y optimización adaptativa de semáforos en Medellín, Colombia.

---

## ¿Qué hace SIPO?

SIPO cruza cuatro fuentes de datos por zona y franja horaria para predecir
el nivel de congestión vial (fluido / moderado / crítico) y genera recomendaciones
operativas para agentes de tránsito en tiempo real.

Tiene dos componentes:

1. **Modelo predictivo** — Red neuronal en PyTorch entrenada con datos históricos
   de incidentes viales, clima y flujo vehicular de Medellín.

2. **Agente RL** — Agente PPO (Stable-Baselines3) que ajusta dinámicamente los
   tiempos de semáforos en una simulación SUMO según las predicciones del modelo.

---

## Demo

```bash
streamlit run dashboard/app.py
```

---

## Estructura del proyecto

```
SIPO/
├── Datasets/                          # Datos históricos de entrenamiento
│   ├── incidentes_viales_medellin.csv
│   ├── precipitacion_medellin.csv
│   ├── temperatura_medellin.csv
│   └── flujo_vehicular_medellin.csv
│
├── data/simulados/                    # Datos simulados para el demo
│   ├── sim_clima.csv
│   ├── sim_eventos.csv
│   ├── sim_flujo_vehicular.csv
│   └── sim_incidentes_activos.csv
│
├── model/
│   ├── model.py                       # Arquitectura de la red neuronal
│   ├── train_model.py                 # Script de entrenamiento (Colab)
│   ├── sipom_model.pt                 # Modelo entrenado
│   └── zona_id_map.json               # Mapa de zonas
│
├── services/
│   ├── clima_service.py
│   ├── eventos_service.py
│   ├── flujo_service.py
│   └── incidentes_service.py
│
├── sumo/
│   ├── network.net.xml                # Red vial simulada
│   ├── routes.rou.xml                 # Rutas de vehículos
│   ├── rl_agent.py                    # Agente PPO
│   └── sipom_ppo.zip                  # Agente entrenado
│
├── notebooks/
│   └── EDA_y_entrenamiento.ipynb      # EDA y entrenamiento en Colab
│
└── dashboard/
    └── app.py                         # Dashboard Streamlit
```

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ |
| Deep Learning | PyTorch |
| Reinforcement Learning | Stable-Baselines3 (PPO) |
| Simulación de tráfico | SUMO + TraCI |
| Dashboard | Streamlit |
| Mapas | Folium + Streamlit-Folium |
| Visualización | Plotly |
| Entrenamiento | Google Colab (GPU T4) |

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/AndresDiaz10202/SIPO.git
cd SIPO

# 2. Instalar dependencias
pip install torch pandas numpy streamlit folium streamlit-folium plotly stable-baselines3 traci

# 3. Generar CSV simulados
python data/simulados/generar_simulados.py

# 4. Ejecutar el dashboard
streamlit run dashboard/app.py
```

> **Nota:** Para ejecutar el agente RL se requiere tener SUMO instalado.
> Descarga: https://sumo.dlr.de/docs/Downloads.php

---

## Modelo predictivo

**Arquitectura:**
```
Input (9 features):
  zona_id        → Embedding(10, 8)
  hora           → int (0-23)
  dia_semana     → int (0-6)
  mes            → int (1-12)
  temperatura    → float (°C)
  lluvia         → int (0/1)
  velocidad_kmh  → float
  volumen_vehiculos → int
  incidentes_hora   → int

Red:
  Dense 256 → BatchNorm → ReLU → Dropout 0.3
  Dense 128 → BatchNorm → ReLU → Dropout 0.3
  Dense 64  → BatchNorm → ReLU
  Output: 3 clases (fluido / moderado / crítico)

Accuracy en test: ~94%
```

**Entrenamiento:**
- Dataset: cruce de 4 fuentes históricas (~200K registros)
- Split: 70% train / 15% validación / 15% test
- Épocas: 30 con ReduceLROnPlateau
- Entorno: Google Colab GPU T4

---

## Zonas SIPO

| Zona | Comunas |
|---|---|
| Av. 80 | 12, 13 |
| Av. Regional | 15, 16 |
| Calle 33 | 10, 11 |
| El Poblado | 14 |
| Guayabal | 15 |
| Laureles | 11 |
| Estadio/Atanasio | 11 |
| La Macarena | 10 |
| Itagüí | 15, 16 |
| Bello | 4, 5 |

---

## Fuentes de datos

| Dataset | Fuente | Período |
|---|---|---|
| Incidentes viales | Secretaría de Movilidad de Medellín | 2015–2024 |
| Precipitación | IDEAM | 2018–2024 |
| Temperatura | IDEAM | 2020–2024 |
| Flujo vehicular | Simulado con patrones reales SIMM | — |

---

## Escalabilidad a producción

En una implementación real los CSV simulados se reemplazarían por:

| CSV simulado | Fuente real |
|---|---|
| `sim_clima.csv` | API IDEAM / OpenWeatherMap |
| `sim_eventos.csv` | Archivo gestionado por la Alcaldía |
| `sim_flujo_vehicular.csv` | Sensores y cámaras SIMM en tiempo real |
| `sim_incidentes_activos.csv` | Operadores de tránsito en vivo |

La arquitectura está diseñada para esa integración sin cambiar el modelo ni el dashboard.

---

## Impacto esperado

SIPO está diseñado para ser adoptado por la Secretaría de Movilidad de Medellín
como herramienta de apoyo a la toma de decisiones operativas, con potencial de
integración futura al sistema SIMTRÁFICO de la ciudad.

---

## Autor

Desarrollado como proyecto de inteligencia artificial aplicada a movilidad urbana.
Medellín, Colombia — 2025.