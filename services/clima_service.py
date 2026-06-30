import pandas as pd
import os

RUTA = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulados', 'sim_clima.csv')


def get_clima(fecha: str, hora: int) -> dict:
    """
    Retorna temperatura y lluvia para una fecha y hora exactas.
    fecha: 'YYYY-MM-DD'
    hora:  int (0-23)
    """
    df = pd.read_csv(RUTA, encoding='utf-8')

    fila = df[(df['fecha'] == fecha) & (df['hora'] == hora)]

    if fila.empty:
        return {
            'temperatura': round(df['temperatura'].mean(), 1),
            'lluvia':      int(df['lluvia'].mode()[0]),
            'mm_lluvia':   round(df['mm_lluvia'].mean(), 1),
        }

    return {
        'temperatura': float(fila['temperatura'].values[0]),
        'lluvia':      int(fila['lluvia'].values[0]),
        'mm_lluvia':   float(fila['mm_lluvia'].values[0]),
    }


def get_clima_dia(fecha: str) -> pd.DataFrame:
    """
    Retorna las 24 horas de clima de un día completo.
    """
    df = pd.read_csv(RUTA, encoding='utf-8')
    resultado = df[df['fecha'] == fecha].sort_values('hora')
    return resultado.reset_index(drop=True)