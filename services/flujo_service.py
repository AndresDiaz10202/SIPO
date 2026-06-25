import pandas as pd
import os

RUTA = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulados', 'sim_flujo_vehicular.csv')


def get_flujo(zona: str, fecha: str, hora: int) -> dict:
    """
    Retorna velocidad y volumen vehicular para una zona, fecha y hora.
    """
    df = pd.read_csv(RUTA, encoding='utf-8')

    fila = df[(df['zona'] == zona) & (df['fecha'] == fecha) & (df['hora'] == hora)]

    if fila.empty:
        # Fallback: promedio de esa zona a esa hora en cualquier día
        fila = df[(df['zona'] == zona) & (df['hora'] == hora)]
        if fila.empty:
            return {'velocidad_kmh': 35.0, 'volumen_vehiculos': 1000, 'flujo_promedio': 'medio'}

    return {
        'velocidad_kmh':    round(float(fila['velocidad_kmh'].mean()), 1),
        'volumen_vehiculos': int(fila['volumen_vehiculos'].mean()),
        'flujo_promedio':    str(fila['flujo_promedio'].mode()[0]),
    }


def get_flujo_todas_zonas(fecha: str, hora: int) -> pd.DataFrame:
    """
    Retorna flujo de todas las zonas para una fecha y hora.
    Útil para el mapa de calor del dashboard.
    """
    df = pd.read_csv(RUTA, encoding='utf-8')

    resultado = df[(df['fecha'] == fecha) & (df['hora'] == hora)].copy()

    if resultado.empty:
        resultado = df[df['hora'] == hora].groupby('zona').mean(numeric_only=True).reset_index()

    return resultado.reset_index(drop=True)