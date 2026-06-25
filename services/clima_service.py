import pandas as pd
import os

RUTA = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulados', 'sim_clima.csv')


def get_clima(fecha: str, hora: int) -> dict:
    """
    Retorna temperatura y lluvia para una fecha y hora dadas.
    fecha: 'YYYY-MM-DD'
    hora:  int (0-23)
    """
    df = pd.read_csv(RUTA, encoding='utf-8')

    # Buscar la franja horaria más cercana a la hora pedida
    df['diff'] = (df['hora'] - hora).abs()
    fila = df[df['fecha'] == fecha].nsmallest(1, 'diff')

    if fila.empty:
        # Si no hay datos para esa fecha usar promedios generales
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
    Retorna todas las franjas horarias de un día completo.
    """
    df = pd.read_csv(RUTA, encoding='utf-8')
    resultado = df[df['fecha'] == fecha].copy()
    return resultado.reset_index(drop=True)