import pandas as pd
import os

RUTA = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulados', 'sim_incidentes_activos.csv')


def get_incidentes_zona(zona: str, fecha: str, hora: int) -> int:
    """
    Retorna el conteo de incidentes activos en una zona para una fecha y hora.
    """
    df = pd.read_csv(RUTA, encoding='utf-8')

    mask = (
        (df['zona']  == zona)  &
        (df['hora']  == hora)  &
        (df['activo'] == 1)    &
        (df['timestamp'].str.startswith(fecha))
    )

    return int(df[mask].shape[0])


def get_incidentes_dia(fecha: str) -> pd.DataFrame:
    """
    Retorna todos los incidentes activos de un día completo.
    """
    df = pd.read_csv(RUTA, encoding='utf-8')
    return df[
        (df['timestamp'].str.startswith(fecha)) &
        (df['activo'] == 1)
    ].reset_index(drop=True)


def get_resumen_incidentes(fecha: str) -> dict:
    """
    Retorna un resumen de incidentes del día por gravedad.
    """
    df = get_incidentes_dia(fecha)

    if df.empty:
        return {'total': 0, 'Solo daños': 0, 'Con heridos': 0, 'Con muertos': 0}

    return {
        'total':        len(df),
        'Solo daños':   int((df['gravedad'] == 'Solo daños').sum()),
        'Con heridos':  int((df['gravedad'] == 'Con heridos').sum()),
        'Con muertos':  int((df['gravedad'] == 'Con muertos').sum()),
    }