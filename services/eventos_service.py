import pandas as pd
import os

RUTA = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulados', 'sim_eventos.csv')


def get_eventos_zona(zona: str, fecha: str, hora: int) -> list:
    """
    Retorna lista de eventos activos en una zona para una fecha y hora dadas.
    """
    df = pd.read_csv(RUTA, encoding='utf-8')

    mask = (
        (df['zona']        == zona)  &
        (df['fecha']       == fecha) &
        (df['hora_inicio'] <= hora)  &
        (df['hora_fin']    >= hora)
    )

    eventos = df[mask].to_dict(orient='records')
    return eventos


def get_eventos_dia(fecha: str) -> pd.DataFrame:
    """
    Retorna todos los eventos de un día, todas las zonas.
    """
    df = pd.read_csv(RUTA, encoding='utf-8')
    return df[df['fecha'] == fecha].reset_index(drop=True)


def hay_evento_masivo(zona: str, fecha: str, hora: int) -> bool:
    """
    Retorna True si hay un evento de impacto alto o medio en esa zona/hora.
    """
    eventos = get_eventos_zona(zona, fecha, hora)
    return any(e['impacto'] in ('alto', 'medio') for e in eventos)