import torch
import torch.nn as nn


class SIPOMModel(nn.Module):
    """
    Red neuronal para predecir nivel de congestión vial en Medellín.
    Entrada : zona_id (embedding) + 8 features numéricas
    Salida  : 3 clases → 0=fluido / 1=moderado / 2=crítico
    """

    def __init__(self, n_zonas: int = 10, emb_dim: int = 8):
        super().__init__()

        # Embedding: convierte el índice de zona en un vector denso
        self.embedding = nn.Embedding(n_zonas, emb_dim)

        # Entrada = embedding (8) + 8 features numéricas = 16
        entrada = emb_dim + 8

        self.red = nn.Sequential(
            # Bloque 1
            nn.Linear(entrada, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Bloque 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Bloque 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Salida
            nn.Linear(64, 3),
        )

    def forward(self, zona_id: torch.Tensor, features_num: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(zona_id)                  # (batch, emb_dim)
        x   = torch.cat([emb, features_num], dim=1)   # (batch, 16)
        return self.red(x)                             # (batch, 3)


# ── Constantes compartidas con el dashboard y los services ──

ZONAS_SIPOM = [
    'Av. 80',
    'Av. Regional',
    'Calle 33',
    'El Poblado',
    'Guayabal',
    'Laureles',
    'Estadio/Atanasio',
    'La Macarena',
    'Itagüí',
    'Bello',
]

ZONA_ID_MAP = {zona: idx for idx, zona in enumerate(ZONAS_SIPOM)}

ETIQUETAS = {
    0: 'Fluido',
    1: 'Moderado',
    2: 'Crítico',
}

COLORES = {
    0: '#2ecc71',   # verde
    1: '#f39c12',   # naranja
    2: '#e74c3c',   # rojo
}


def cargar_modelo(ruta_pt: str, device: str = 'cpu') -> SIPOMModel:
    """
    Carga el modelo entrenado desde el archivo .pt y lo pone en modo evaluación.
    Uso:
        from model.model import cargar_modelo
        modelo = cargar_modelo('model/sipom_model.pt')
    """
    modelo = SIPOMModel()
    modelo.load_state_dict(torch.load(ruta_pt, map_location=device))
    modelo.eval()
    return modelo


def predecir(modelo: SIPOMModel, zona: str, hora: int, dia_semana: int,
             mes: int, temperatura: float, lluvia: int,
             velocidad_kmh: float, volumen_vehiculos: int,
             incidentes_hora: int, device: str = 'cpu') -> dict:
    """
    Hace una predicción para una zona y franja horaria.
    Retorna dict con clase, etiqueta, color, probabilidades y factor principal.
    """
    zona_id = ZONA_ID_MAP.get(zona, 0)

    t_zona = torch.tensor([zona_id], dtype=torch.long).to(device)
    t_num  = torch.tensor([[
        hora, dia_semana, mes,
        temperatura, lluvia,
        velocidad_kmh, volumen_vehiculos, incidentes_hora
    ]], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = modelo(t_zona, t_num)
        probs  = torch.softmax(logits, dim=1).squeeze().tolist()
        clase  = int(torch.argmax(logits, dim=1).item())

    factor_principal = _factor_dominante(
        velocidad_kmh, volumen_vehiculos, incidentes_hora, lluvia, hora
    )

    return {
        'clase':         clase,
        'etiqueta':      ETIQUETAS[clase],
        'color':         COLORES[clase],
        'probabilidades': {
            'Fluido':   round(probs[0], 3),
            'Moderado': round(probs[1], 3),
            'Crítico':  round(probs[2], 3),
        },
        'factor_principal': factor_principal,
    }


def _factor_dominante(velocidad_kmh, volumen_vehiculos, incidentes_hora, lluvia, hora) -> str:
    """
    Aproximación simple e interpretable de qué factor pesa más en
    el contexto actual, basada en umbrales de referencia.
    No es una atribución exacta del modelo, es una guía orientativa.
    """
    factores = []

    if velocidad_kmh < 20:
        factores.append(('Velocidad muy baja', 3))
    elif velocidad_kmh < 30:
        factores.append(('Velocidad reducida', 2))

    if volumen_vehiculos > 1800:
        factores.append(('Volumen vehicular alto', 3))
    elif volumen_vehiculos > 1200:
        factores.append(('Volumen vehicular moderado', 2))

    if incidentes_hora > 0:
        factores.append(('Incidentes activos en la zona', 3 + incidentes_hora))

    if lluvia:
        factores.append(('Lluvia activa', 2))

    if 7 <= hora <= 9 or 17 <= hora <= 19:
        factores.append(('Hora pico', 2))

    if not factores:
        return 'Condiciones normales — sin factores de riesgo destacados'

    factores.sort(key=lambda x: x[1], reverse=True)
    return factores[0][0]