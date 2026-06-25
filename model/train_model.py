"""
train_model.py — Script de entrenamiento de SIPOM
Se sube a Google Colab y se ejecuta allá con GPU.
NO se ejecuta en VS Code.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Rutas (ajustar si es necesario en Colab) ──
RUTA_DATASETS     = '/content/drive/MyDrive/SIPO/Datasets'
RUTA_MODELO_OUT   = '/content/drive/MyDrive/SIPO/sipom_model.pt'
RUTA_ZONAS_OUT    = '/content/drive/MyDrive/SIPO/zona_id_map.json'

RUTA_INCIDENTES    = os.path.join(RUTA_DATASETS, 'incidentes_viales_medellin.csv')
RUTA_PRECIPITACION = os.path.join(RUTA_DATASETS, 'precipitacion_medellin.csv')
RUTA_TEMPERATURA   = os.path.join(RUTA_DATASETS, 'temperatura_medellin.csv')
RUTA_FLUJO         = os.path.join(RUTA_DATASETS, 'flujo_vehicular_medellin.csv')

# ── Constantes ──
ZONAS_SIPOM = [
    'Av. 80', 'Av. Regional', 'Calle 33', 'El Poblado', 'Guayabal',
    'Laureles', 'Estadio/Atanasio', 'La Macarena', 'Itagüí', 'Bello',
]
MAPA_ZONA_ID = {zona: idx for idx, zona in enumerate(ZONAS_SIPOM)}

MAPA_COMUNA_ZONA = {
    4: 'Bello', 5: 'Bello',
    10: 'La Macarena', 11: 'Laureles',
    12: 'Av. 80', 13: 'Av. 80',
    14: 'El Poblado',
    15: 'Av. Regional', 16: 'Av. Regional',
}

MAPA_TARGET = {'bajo': 0, 'medio': 1, 'alto': 2, 'muy_alto': 2}

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
EPOCHS     = 30
LR         = 1e-3


# ──────────────────────────────────────────────
# 1. Carga y preprocesamiento
# ──────────────────────────────────────────────

def cargar_flujo():
    df = pd.read_csv(RUTA_FLUJO, encoding='utf-8', low_memory=False)
    df['fecha_hora'] = pd.to_datetime(
        df['fecha'].astype(str) + ' ' + df['hora'].astype(str), errors='coerce'
    ).dt.floor('h')
    df['hora_int']       = df['fecha_hora'].dt.hour
    df['dia_semana_int'] = df['fecha_hora'].dt.dayofweek
    df['mes_int']        = df['fecha_hora'].dt.month
    return df[['fecha_hora', 'zona', 'hora_int', 'dia_semana_int',
               'mes_int', 'velocidad_kmh', 'volumen_vehiculos', 'flujo_promedio']].copy()


def cargar_clima():
    # Temperatura
    df_t = pd.read_csv(RUTA_TEMPERATURA, encoding='latin-1', low_memory=False)
    df_t['ValorObservado'] = (
        df_t['ValorObservado'].astype(str)
        .str.replace(',', '.', regex=False)
        .pipe(pd.to_numeric, errors='coerce')
    )
    df_t['fecha_dt'] = pd.to_datetime(
        df_t['FechaObservacion'], format='%Y %b %d %I:%M:%S %p', errors='coerce'
    )
    df_temp_h = (
        df_t.set_index('fecha_dt')['ValorObservado']
        .resample('h').mean().reset_index()
        .rename(columns={'ValorObservado': 'temperatura', 'fecha_dt': 'fecha_hora'})
    )

    # Precipitación
    df_p = pd.read_csv(RUTA_PRECIPITACION, encoding='latin-1', low_memory=False)
    df_p['ValorObservado'] = (
        df_p['ValorObservado'].astype(str)
        .str.replace(',', '.', regex=False)
        .pipe(pd.to_numeric, errors='coerce')
    )
    df_p['fecha_dt'] = pd.to_datetime(
        df_p['FechaObservacion'], format='%Y %b %d %I:%M:%S %p', errors='coerce'
    )
    df_prec_h = (
        df_p.set_index('fecha_dt')['ValorObservado']
        .resample('h').sum().reset_index()
        .rename(columns={'ValorObservado': 'mm_lluvia', 'fecha_dt': 'fecha_hora'})
    )
    df_prec_h['lluvia'] = (df_prec_h['mm_lluvia'] > 0.5).astype(int)

    df_clima = pd.merge(df_temp_h, df_prec_h, on='fecha_hora', how='outer')
    df_clima['fecha_hora'] = df_clima['fecha_hora'].dt.floor('h')
    mediana = df_clima['temperatura'].median()
    df_clima['temperatura'] = df_clima['temperatura'].fillna(mediana)
    df_clima['lluvia']      = df_clima['lluvia'].fillna(0).astype(int)
    return df_clima, mediana


def cargar_incidentes():
    df = pd.read_csv(RUTA_INCIDENTES, encoding='latin-1', low_memory=False)
    df['fecha_dt'] = pd.to_datetime(
        df['FECHA_ACCIDENTE'], format='%d/%m/%Y %H:%M:%S', errors='coerce'
    )
    if df['fecha_dt'].isnull().sum() > len(df) * 0.1:
        df['fecha_dt'] = pd.to_datetime(df['FECHA_ACCIDENTES'], errors='coerce')
    df['zona']       = df['NUMCOMUNA'].map(MAPA_COMUNA_ZONA)
    df['fecha_hora'] = df['fecha_dt'].dt.floor('h')
    return (
        df.dropna(subset=['zona', 'fecha_hora'])
        .groupby(['zona', 'fecha_hora']).size()
        .reset_index(name='incidentes_hora')
    )


def construir_dataset():
    print('Cargando datasets...')
    df_flujo              = cargar_flujo()
    df_clima, mediana     = cargar_clima()
    df_inc_zh             = cargar_incidentes()

    df = pd.merge(df_flujo,  df_clima,  on='fecha_hora',          how='left')
    df = pd.merge(df,        df_inc_zh, on=['zona', 'fecha_hora'], how='left')

    df['incidentes_hora'] = df['incidentes_hora'].fillna(0).astype(int)
    df['temperatura']     = df['temperatura'].fillna(mediana)
    df['lluvia']          = df['lluvia'].fillna(0).astype(int)
    df['target']          = df['flujo_promedio'].map(MAPA_TARGET)
    df['zona_id']         = df['zona'].map(MAPA_ZONA_ID)

    # Ruido para evitar memorización
    np.random.seed(42)
    df['temperatura']      += np.random.normal(0, 1.5, len(df))
    df['velocidad_kmh']    += np.random.normal(0, 3.0, len(df))
    df['volumen_vehiculos'] = (
        df['volumen_vehiculos'] * np.random.uniform(0.85, 1.15, len(df))
    ).astype(int)
    idx = np.random.choice(len(df), size=int(len(df) * 0.08), replace=False)
    df.loc[idx, 'target'] = np.random.randint(0, 3, size=len(idx))

    FEATURES = [
        'zona_id', 'hora_int', 'dia_semana_int', 'mes_int',
        'temperatura', 'lluvia', 'velocidad_kmh', 'volumen_vehiculos', 'incidentes_hora',
    ]
    df = df[FEATURES + ['target']].dropna().reset_index(drop=True)
    for col in ['zona_id', 'hora_int', 'dia_semana_int', 'mes_int',
                'lluvia', 'incidentes_hora', 'target']:
        df[col] = df[col].astype(int)

    print(f'Dataset listo: {len(df):,} filas')
    return df


# ──────────────────────────────────────────────
# 2. Dataset PyTorch
# ──────────────────────────────────────────────

class SIPOMDataset(Dataset):
    COLS_NUM = ['hora_int', 'dia_semana_int', 'mes_int',
                'temperatura', 'lluvia', 'velocidad_kmh',
                'volumen_vehiculos', 'incidentes_hora']

    def __init__(self, df):
        self.zona_id  = torch.tensor(df['zona_id'].values, dtype=torch.long)
        self.features = torch.tensor(df[self.COLS_NUM].values, dtype=torch.float32)
        self.target   = torch.tensor(df['target'].values, dtype=torch.long)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.zona_id[idx], self.features[idx], self.target[idx]


# ──────────────────────────────────────────────
# 3. Modelo
# ──────────────────────────────────────────────

class SIPOMModel(nn.Module):
    def __init__(self, n_zonas=10, emb_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(n_zonas, emb_dim)
        entrada = emb_dim + 8
        self.red = nn.Sequential(
            nn.Linear(entrada, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, zona_id, features_num):
        emb = self.embedding(zona_id)
        x   = torch.cat([emb, features_num], dim=1)
        return self.red(x)


# ──────────────────────────────────────────────
# 4. Entrenamiento
# ──────────────────────────────────────────────

def ejecutar_epoca(modelo, loader, criterio, optimizador=None):
    entrenando = optimizador is not None
    modelo.train() if entrenando else modelo.eval()
    perdida_total, correctos, total = 0, 0, 0

    ctx = torch.enable_grad() if entrenando else torch.no_grad()
    with ctx:
        for zona_id, features, target in loader:
            zona_id, features, target = (
                zona_id.to(DEVICE), features.to(DEVICE), target.to(DEVICE)
            )
            salida  = modelo(zona_id, features)
            perdida = criterio(salida, target)
            if entrenando:
                optimizador.zero_grad()
                perdida.backward()
                optimizador.step()
            perdida_total += perdida.item() * len(target)
            correctos     += (salida.argmax(dim=1) == target).sum().item()
            total         += len(target)

    return perdida_total / total, correctos / total


def entrenar():
    df = construir_dataset()

    df_temp_s, df_test_s = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df['target']
    )
    df_tr, df_val = train_test_split(
        df_temp_s, test_size=0.176, random_state=42, stratify=df_temp_s['target']
    )

    dl_train = DataLoader(SIPOMDataset(df_tr),    batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    dl_val   = DataLoader(SIPOMDataset(df_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    dl_test  = DataLoader(SIPOMDataset(df_test_s),batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    modelo      = SIPOMModel().to(DEVICE)
    criterio    = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode='min', patience=5, factor=0.5
    )

    mejor_val_loss = float('inf')
    print(f'Entrenando en {DEVICE}...\n')

    for epoca in range(1, EPOCHS + 1):
        tr_loss, tr_acc = ejecutar_epoca(modelo, dl_train, criterio, optimizador)
        vl_loss, vl_acc = ejecutar_epoca(modelo, dl_val,   criterio)
        scheduler.step(vl_loss)

        if vl_loss < mejor_val_loss:
            mejor_val_loss = vl_loss
            torch.save(modelo.state_dict(), RUTA_MODELO_OUT)
            marca = ' ← mejor'
        else:
            marca = ''

        print(
            f'Época {epoca:02d}/{EPOCHS} | '
            f'Train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | '
            f'Val loss: {vl_loss:.4f} acc: {vl_acc:.3f}{marca}'
        )

    # Evaluación final en test
    modelo.load_state_dict(torch.load(RUTA_MODELO_OUT, map_location=DEVICE))
    modelo.eval()
    preds, reales = [], []
    with torch.no_grad():
        for zona_id, features, target in dl_test:
            zona_id, features = zona_id.to(DEVICE), features.to(DEVICE)
            preds.extend(modelo(zona_id, features).argmax(dim=1).cpu().numpy())
            reales.extend(target.numpy())

    print('\n=== Reporte final ===')
    print(classification_report(reales, preds, target_names=['Fluido', 'Moderado', 'Crítico']))

    # Guardar mapa de zonas
    with open(RUTA_ZONAS_OUT, 'w', encoding='utf-8') as f:
        json.dump(MAPA_ZONA_ID, f, ensure_ascii=False, indent=2)

    print(f'✅ Modelo guardado en: {RUTA_MODELO_OUT}')
    print(f'✅ Mapa zonas en:      {RUTA_ZONAS_OUT}')


if __name__ == '__main__':
    entrenar()