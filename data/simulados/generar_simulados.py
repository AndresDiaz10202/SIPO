"""
generar_simulados.py
Genera los 4 CSV simulados para el demo del dashboard.
Ejecutar UNA sola vez desde VS Code:
    cd C:\\Users\\SSSA\\Downloads\\Andres\\SIPO
    python data/simulados/generar_simulados.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

RUTA = os.path.join(os.path.dirname(__file__))

ZONAS = [
    'Av. 80', 'Av. Regional', 'Calle 33', 'El Poblado', 'Guayabal',
    'Laureles', 'Estadio/Atanasio', 'La Macarena', 'Itagüí', 'Bello',
]

FRANJAS = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']

HOY   = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
DIAS  = [HOY + timedelta(days=i) for i in range(7)]


# ──────────────────────────────────────────────
# 1. sim_clima.csv
# ──────────────────────────────────────────────

def generar_clima():
    filas = []
    for dia in DIAS:
        for hora in range(24):
            # Temperatura varía por hora: más fría de madrugada, más cálida al mediodía
            temp_base = 18 + 8 * np.sin((hora - 6) * np.pi / 12)
            temp      = round(temp_base + np.random.normal(0, 1.0), 1)

            # Probabilidad de lluvia por hora — más alta en la tarde en Medellín
            prob_lluvia_hora = {
                0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.08,
                6: 0.10, 7: 0.10, 8: 0.12, 9: 0.15, 10: 0.18, 11: 0.22,
                12: 0.28, 13: 0.35, 14: 0.50, 15: 0.60, 16: 0.62,
                17: 0.55, 18: 0.40, 19: 0.25, 20: 0.15, 21: 0.10,
                22: 0.08, 23: 0.06,
            }
            lluvia    = int(np.random.random() < prob_lluvia_hora[hora])
            mm_lluvia = round(np.random.uniform(1.0, 25.0), 1) if lluvia else 0.0

            filas.append({
                'fecha':       dia.strftime('%Y-%m-%d'),
                'hora':        hora,
                'dia_semana':  dia.weekday(),
                'temperatura': temp,
                'lluvia':      lluvia,
                'mm_lluvia':   mm_lluvia,
            })

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(RUTA, 'sim_clima.csv'), index=False, encoding='utf-8')
    print(f'✅ sim_clima.csv — {len(df)} filas')


# ──────────────────────────────────────────────
# 2. sim_eventos.csv
# ──────────────────────────────────────────────

def generar_eventos():
    eventos = [
        ('Partido Atlético Nacional',    'Estadio/Atanasio', 45000, 1),
        ('Concierto Parque Explora',     'La Macarena',      8000,  2),
        ('Feria Exposición',             'Guayabal',         20000, 3),
        ('Partido Independiente',        'Estadio/Atanasio', 38000, 4),
        ('Marcha Centro',                'Calle 33',         15000, 0),
        ('Festival Flores El Poblado',   'El Poblado',       12000, 5),
        ('Evento Bello Centro',          'Bello',            6000,  6),
        ('Feria Laureles',               'Laureles',         5000,  1),
    ]

    filas = []
    for nombre, zona, aforo, dia_offset in eventos:
        dia        = HOY + timedelta(days=dia_offset)
        hora_inicio = np.random.choice([10, 14, 16, 18, 20])
        hora_fin    = hora_inicio + np.random.choice([2, 3, 4])

        filas.append({
            'nombre':      nombre,
            'zona':        zona,
            'fecha':       dia.strftime('%Y-%m-%d'),
            'hora_inicio': hora_inicio,
            'hora_fin':    min(hora_fin, 23),
            'aforo':       aforo,
            'impacto':     'alto' if aforo > 20000 else 'medio' if aforo > 8000 else 'bajo',
        })

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(RUTA, 'sim_eventos.csv'), index=False, encoding='utf-8')
    print(f'✅ sim_eventos.csv — {len(df)} filas')


# ──────────────────────────────────────────────
# 3. sim_flujo_vehicular.csv
# ──────────────────────────────────────────────

def generar_flujo():
    # Velocidades base por zona (km/h en condiciones normales)
    velocidad_base = {
        'Av. 80':            45, 'Av. Regional':      55,
        'Calle 33':          35, 'El Poblado':        40,
        'Guayabal':          50, 'Laureles':          38,
        'Estadio/Atanasio':  30, 'La Macarena':       32,
        'Itagüí':            48, 'Bello':             50,
    }
    # Volúmenes base por zona (vehículos/hora)
    volumen_base = {
        'Av. 80':            1800, 'Av. Regional':      2200,
        'Calle 33':          1200, 'El Poblado':        1500,
        'Guayabal':          1600, 'Laureles':          1100,
        'Estadio/Atanasio':  900,  'La Macarena':       800,
        'Itagüí':            1700, 'Bello':             1400,
    }

    filas = []
    for dia in DIAS:
        for hora in range(24):
            # Factor de hora pico: mañana 7-9h y tarde 17-19h
            # Curva horaria más granular — pico real de Medellín
            curva_hora = {
                0: 0.20, 1: 0.15, 2: 0.12, 3: 0.10, 4: 0.15, 5: 0.35,
                6: 0.75, 7: 1.55, 8: 1.70, 9: 1.30, 10: 1.00, 11: 0.95,
                12: 1.10, 13: 1.05, 14: 0.90, 15: 0.95, 16: 1.15,
                17: 1.50, 18: 1.75, 19: 1.45, 20: 1.05, 21: 0.70,
                22: 0.45, 23: 0.30,
            }
            factor_hora = curva_hora[hora] * np.random.uniform(0.92, 1.08)

            # Factor fin de semana
            factor_dia = 0.65 if dia.weekday() >= 5 else 1.0

            for zona in ZONAS:
                vel = velocidad_base[zona] / factor_hora * factor_dia
                vel = round(vel + np.random.normal(0, 3), 1)
                vol = int(volumen_base[zona] * factor_hora * factor_dia
                          + np.random.randint(-100, 100))

                if vel < 15:
                    flujo = 'muy_alto'
                elif vel < 25:
                    flujo = 'alto'
                elif vel < 38:
                    flujo = 'medio'
                else:
                    flujo = 'bajo'

                filas.append({
                    'fecha':             dia.strftime('%Y-%m-%d'),
                    'hora':              hora,
                    'dia_semana':        dia.weekday(),
                    'zona':              zona,
                    'velocidad_kmh':     max(vel, 5.0),
                    'volumen_vehiculos': max(vol, 0),
                    'flujo_promedio':    flujo,
                })

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(RUTA, 'sim_flujo_vehicular.csv'), index=False, encoding='utf-8')
    print(f'✅ sim_flujo_vehicular.csv — {len(df)} filas')


# ──────────────────────────────────────────────
# 4. sim_incidentes_activos.csv
# ──────────────────────────────────────────────

def generar_incidentes():
    tipos   = ['Choque', 'Atropello', 'Volcamiento', 'Caída Ocupante', 'Otro']
    gravedades = ['Solo daños', 'Con heridos', 'Con muertos']
    pesos_grav = [0.70, 0.25, 0.05]

    filas = []
    for dia in DIAS:
        # Entre 3 y 8 incidentes por día
        n_inc = np.random.randint(3, 9)
        for _ in range(n_inc):
            hora   = np.random.randint(0, 24)
            minuto = np.random.randint(0, 60)
            zona   = np.random.choice(ZONAS)
            ts     = dia.replace(hour=hora, minute=minuto)

            filas.append({
                'timestamp':  ts.strftime('%Y-%m-%d %H:%M:%S'),
                'zona':       zona,
                'tipo':       np.random.choice(tipos),
                'gravedad':   np.random.choice(gravedades, p=pesos_grav),
                'activo':     1,
                'hora':       hora,
            })

    df = pd.DataFrame(filas).sort_values('timestamp').reset_index(drop=True)
    df.to_csv(os.path.join(RUTA, 'sim_incidentes_activos.csv'), index=False, encoding='utf-8')
    print(f'✅ sim_incidentes_activos.csv — {len(df)} filas')


# ──────────────────────────────────────────────
# Ejecutar todo
# ──────────────────────────────────────────────

if __name__ == '__main__':
    generar_clima()
    generar_eventos()
    generar_flujo()
    generar_incidentes()
    print('\n✅ Los 4 CSV simulados están listos en data/simulados/')