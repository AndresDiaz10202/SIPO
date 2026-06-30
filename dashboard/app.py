"""
app.py — Dashboard principal de SIPOM
Ejecutar con: streamlit run dashboard/app.py
"""

import sys
import os

# Agregar la raíz del proyecto al path para importar model/ y services/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from datetime import datetime

from model.model import cargar_modelo, predecir, ZONAS_SIPOM, ETIQUETAS, COLORES
from services.clima_service      import get_clima, get_clima_dia
from services.eventos_service    import get_eventos_dia, hay_evento_masivo
from services.flujo_service      import get_flujo, get_flujo_todas_zonas
from services.incidentes_service import get_incidentes_zona, get_incidentes_dia, get_resumen_incidentes

# ── Coordenadas de cada zona para el mapa ──
COORDENADAS = {
    'Av. 80':            (6.2520, -75.5950),
    'Av. Regional':      (6.2200, -75.5780),
    'Calle 33':          (6.2280, -75.5650),
    'El Poblado':        (6.2080, -75.5680),
    'Guayabal':          (6.2230, -75.5870),
    'Laureles':          (6.2440, -75.5970),
    'Estadio/Atanasio':  (6.2540, -75.5910),
    'La Macarena':       (6.2495, -75.5805),
    'Itagüí':            (6.1850, -75.5990),
    'Bello':             (6.3350, -75.5570),
}

RUTA_MODELO = os.path.join(os.path.dirname(__file__), '..', 'model', 'sipom_model.pt')


# ──────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────

st.set_page_config(
    page_title='SIPOM — Movilidad Medellín',
    page_icon='🚦',
    layout='wide',
)

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; }
        .metric-card {
            background: #1e1e2e;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Carga del modelo (una sola vez con cache)
# ──────────────────────────────────────────────

@st.cache_resource
def cargar():
    return cargar_modelo(RUTA_MODELO)

modelo = cargar()


# ──────────────────────────────────────────────
# Sidebar — controles
# ──────────────────────────────────────────────

with st.sidebar:
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/'
             'Coat_of_arms_of_Medell%C3%ADn.svg/200px-Coat_of_arms_of_Medell%C3%ADn.svg.png',
             width=80)
    st.title('SIPOM')
    st.caption('Sistema Inteligente de Predicción y Optimización de Movilidad Urbana')
    st.divider()

    fecha_sel = st.date_input('Fecha', value=datetime.today())
    hora_sel  = st.slider('Hora', min_value=0, max_value=23, value=datetime.now().hour)
    zona_sel  = st.selectbox('Zona', ZONAS_SIPOM)

    st.divider()
    st.caption('Secretaría de Movilidad — Medellín')


fecha_str = fecha_sel.strftime('%Y-%m-%d')
dia_semana = fecha_sel.weekday()
mes        = fecha_sel.month


# ──────────────────────────────────────────────
# Obtener datos de los services
# ──────────────────────────────────────────────

clima      = get_clima(fecha_str, hora_sel)
flujo      = get_flujo(zona_sel, fecha_str, hora_sel)
incidentes = get_incidentes_zona(zona_sel, fecha_str, hora_sel)

resultado = predecir(
    modelo        = modelo,
    zona          = zona_sel,
    hora          = hora_sel,
    dia_semana    = dia_semana,
    mes           = mes,
    temperatura   = clima['temperatura'],
    lluvia        = clima['lluvia'],
    velocidad_kmh = flujo['velocidad_kmh'],
    volumen_vehiculos = flujo['volumen_vehiculos'],
    incidentes_hora   = incidentes,
)


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.title('🚦 SIPOM — Predicción de Congestión Vial')
st.caption(f'Medellín · {fecha_str} · {hora_sel:02d}:00 · Zona: {zona_sel}')
st.divider()


# ──────────────────────────────────────────────
# Fila 1 — Métricas principales
# ──────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    color = resultado['color']
    nivel = resultado['etiqueta']
    st.markdown(f"""
        <div style='background:{color}22; border-left: 4px solid {color};
                    padding:1rem; border-radius:8px;'>
            <h3 style='color:{color}; margin:0'>🚦 {nivel}</h3>
            <p style='margin:0; color:#aaa'>Nivel de congestión</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric('🌡️ Temperatura', f"{clima['temperatura']}°C")

with col3:
    lluvia_txt = '🌧️ Sí' if clima['lluvia'] else '☀️ No'
    st.metric('Lluvia', lluvia_txt)

with col4:
    st.metric('🚗 Velocidad', f"{flujo['velocidad_kmh']} km/h")

with col5:
    st.metric('⚠️ Incidentes', incidentes)

st.divider()

# ──────────────────────────────────────────────
# Fila 2 — Mapa + Probabilidades
# ──────────────────────────────────────────────

col_mapa, col_probs = st.columns([2, 1])

with col_mapa:
    st.subheader('🗺️ Mapa de Congestión por Zona')

    # Predecir todas las zonas para la hora seleccionada
    flujo_todas = get_flujo_todas_zonas(fecha_str, hora_sel)

    m = folium.Map(
        location=[6.2442, -75.5812],
        zoom_start=12,
        tiles='CartoDB dark_matter'
    )

    for zona in ZONAS_SIPOM:
        lat, lon = COORDENADAS[zona]

        # Obtener datos de esa zona
        f = get_flujo(zona, fecha_str, hora_sel)
        c = get_clima(fecha_str, hora_sel)
        i = get_incidentes_zona(zona, fecha_str, hora_sel)

        res = predecir(
            modelo=modelo, zona=zona, hora=hora_sel,
            dia_semana=dia_semana, mes=mes,
            temperatura=c['temperatura'], lluvia=c['lluvia'],
            velocidad_kmh=f['velocidad_kmh'],
            volumen_vehiculos=f['volumen_vehiculos'],
            incidentes_hora=i,
        )

        color_hex = res['color']
        nivel     = res['etiqueta']

        folium.CircleMarker(
            location=[lat, lon],
            radius=18,
            color=color_hex,
            fill=True,
            fill_color=color_hex,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{zona}</b><br>"
                f"Nivel: {nivel}<br>"
                f"Velocidad: {f['velocidad_kmh']} km/h<br>"
                f"Volumen: {f['volumen_vehiculos']} veh/h<br>"
                f"Incidentes: {i}",
                max_width=200
            ),
            tooltip=f"{zona} — {nivel}",
        ).add_to(m)

        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f"<div style='font-size:9px; color:white; "
                     f"font-weight:bold; text-align:center; "
                     f"width:80px; margin-left:-40px;'>{zona}</div>",
                icon_size=(80, 20),
                icon_anchor=(0, 0),
            )
        ).add_to(m)

    st_folium(m, width=700, height=420)

with col_probs:
    st.subheader('📊 Probabilidades')

    probs = resultado['probabilidades']

    fig_probs = go.Figure(go.Bar(
        x=list(probs.values()),
        y=list(probs.keys()),
        orientation='h',
        marker_color=['#2ecc71', '#f39c12', '#e74c3c'],
        text=[f"{v*100:.1f}%" for v in probs.values()],
        textposition='outside',
    ))
    fig_probs.update_layout(
        xaxis=dict(range=[0, 1], showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=200,
        margin=dict(l=10, r=40, t=10, b=10),
    )
    st.plotly_chart(fig_probs, use_container_width=True)

    st.divider()
    st.subheader('📋 Eventos hoy')

    eventos_dia = get_eventos_dia(fecha_str)
    if eventos_dia.empty:
        st.info('Sin eventos programados')
    else:
        for _, ev in eventos_dia.iterrows():
            color_ev = '#e74c3c' if ev['impacto'] == 'alto' else \
                       '#f39c12' if ev['impacto'] == 'medio' else '#2ecc71'
            st.markdown(f"""
                <div style='border-left: 3px solid {color_ev};
                            padding: 0.4rem 0.8rem; margin-bottom:0.4rem;
                            background: {color_ev}11; border-radius:4px;'>
                    <b>{ev['nombre']}</b><br>
                    <small>{ev['zona']} · {ev['hora_inicio']}:00–{ev['hora_fin']}:00
                    · {ev['aforo']:,} personas</small>
                </div>
            """, unsafe_allow_html=True)

st.divider()


# ──────────────────────────────────────────────
# Fila 3 — Evolución horaria + Incidentes del día
# ──────────────────────────────────────────────

col_evol, col_inc = st.columns([2, 1])

with col_evol:
    st.subheader(f'📈 Evolución del día — {zona_sel}')

    horas   = list(range(24))
    niveles = []
    vels    = []
    vols    = []

    for h in horas:
        f_h = get_flujo(zona_sel, fecha_str, h)
        c_h = get_clima(fecha_str, h)
        i_h = get_incidentes_zona(zona_sel, fecha_str, h)
        r_h = predecir(
            modelo=modelo, zona=zona_sel, hora=h,
            dia_semana=dia_semana, mes=mes,
            temperatura=c_h['temperatura'], lluvia=c_h['lluvia'],
            velocidad_kmh=f_h['velocidad_kmh'],
            volumen_vehiculos=f_h['volumen_vehiculos'],
            incidentes_hora=i_h,
        )
        niveles.append(r_h['clase'])
        vels.append(f_h['velocidad_kmh'])
        vols.append(f_h['volumen_vehiculos'])

    df_evol = pd.DataFrame({
        'hora':    horas,
        'nivel':   niveles,
        'velocidad': vels,
        'volumen':   vols,
    })

    colores_hora = [COLORES[n] for n in niveles]

    fig_evol = go.Figure()
    fig_evol.add_trace(go.Bar(
        x=df_evol['hora'],
        y=df_evol['nivel'],
        marker_color=colores_hora,
        name='Nivel congestión',
        hovertemplate='Hora %{x}:00<br>Nivel: %{y}<extra></extra>',
    ))
    fig_evol.add_trace(go.Scatter(
        x=df_evol['hora'],
        y=[v / 10 for v in df_evol['velocidad']],
        mode='lines+markers',
        name='Velocidad / 10',
        line=dict(color='#3498db', width=2),
        hovertemplate='Hora %{x}:00<br>Vel: %{customdata} km/h<extra></extra>',
        customdata=df_evol['velocidad'],
    ))
    fig_evol.add_vline(
        x=hora_sel, line_dash='dash', line_color='white', line_width=1,
        annotation_text=f'{hora_sel:02d}:00', annotation_position='top right'
    )
    fig_evol.update_layout(
        xaxis=dict(title='Hora', tickmode='linear', dtick=2),
        yaxis=dict(title='Nivel (0=fluido, 2=crítico)', range=[-0.2, 2.5]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(orientation='h', y=1.1),
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_evol, use_container_width=True)

with col_inc:
    st.subheader('🚨 Incidentes del día')

    resumen = get_resumen_incidentes(fecha_str)
    st.metric('Total incidentes', resumen['total'])

    col_a, col_b, col_c = st.columns(3)
    col_a.metric('Solo daños',  resumen['Solo daños'])
    col_b.metric('Con heridos', resumen['Con heridos'])
    col_c.metric('Con muertos', resumen['Con muertos'])

    df_inc_dia = get_incidentes_dia(fecha_str)
    if not df_inc_dia.empty:
        fig_inc = px.bar(
            df_inc_dia.groupby('zona').size().reset_index(name='total'),
            x='total', y='zona', orientation='h',
            color='total',
            color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c'],
        )
        fig_inc.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            coloraxis_showscale=False,
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title='', yaxis_title='',
        )
        st.plotly_chart(fig_inc, use_container_width=True)

st.divider()


# ──────────────────────────────────────────────
# Fila 4 — Recomendaciones operativas
# ──────────────────────────────────────────────

st.subheader('📌 Recomendaciones Operativas')

clase   = resultado['clase']
evento  = hay_evento_masivo(zona_sel, fecha_str, hora_sel)
llueve  = clima['lluvia'] == 1

recomendaciones = []

if clase == 2:
    recomendaciones.append(('🔴 CRÍTICO', f'Desplegar agentes de tránsito en {zona_sel} de inmediato.'))
    recomendaciones.append(('🔴 CRÍTICO', 'Activar rutas alternativas y señalización dinámica.'))
elif clase == 1:
    recomendaciones.append(('🟡 MODERADO', f'Monitorear {zona_sel} — posible incremento de congestión.'))
    recomendaciones.append(('🟡 MODERADO', 'Considerar refuerzo de agentes en próximas 2 horas.'))
else:
    recomendaciones.append(('🟢 FLUIDO', f'{zona_sel} opera con normalidad. Sin intervención requerida.'))

if llueve:
    recomendaciones.append(('🌧️ CLIMA', 'Lluvia activa — reducir velocidades recomendadas y aumentar vigilancia.'))

if evento:
    recomendaciones.append(('📅 EVENTO', f'Evento masivo en {zona_sel} — coordinar operativo de tránsito.'))

if incidentes > 0:
    recomendaciones.append(('⚠️ INCIDENTES', f'{incidentes} incidente(s) activo(s) en la zona — verificar estado.'))

cols_rec = st.columns(len(recomendaciones))
for col, (etiq, texto) in zip(cols_rec, recomendaciones):
    color_rec = '#e74c3c' if 'CRÍTICO' in etiq else \
                '#f39c12' if 'MODERADO' in etiq or 'CLIMA' in etiq or \
                             'EVENTO' in etiq or 'INCIDENTE' in etiq else '#2ecc71'
    with col:
        st.markdown(f"""
            <div style='border: 1px solid {color_rec}; border-radius:8px;
                        padding:0.8rem; background:{color_rec}11; height:100%;'>
                <b style='color:{color_rec}'>{etiq}</b><br>
                <small>{texto}</small>
            </div>
        """, unsafe_allow_html=True)

st.divider()
st.caption('SIPOM v1.0 · Secretaría de Movilidad de Medellín · Modelo entrenado con datos históricos 2015–2024')