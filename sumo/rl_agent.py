"""
rl_agent.py — Agente PPO para optimización de semáforos con SUMO
Entrena un agente de Reinforcement Learning que ajusta los tiempos
de semáforos según el nivel de congestión predicho por SIPOM.

Ejecutar desde la raíz del proyecto:
    python sumo/rl_agent.py
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Agregar raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import cargar_modelo, predecir, ZONAS_SIPOM

# ── Rutas SUMO ──
SUMO_HOME   = os.environ.get('SUMO_HOME', 'C:/Program Files (x86)/Eclipse/Sumo')
SUMO_BINARY = os.path.join(SUMO_HOME, 'bin', 'sumo')
NETWORK     = os.path.join(os.path.dirname(__file__), 'network.net.xml')
ROUTES      = os.path.join(os.path.dirname(__file__), 'routes.rou.xml')
RUTA_MODELO = os.path.join(os.path.dirname(__file__), '..', 'model', 'sipom_model.pt')

# Agregar tools de SUMO al path para importar traci
sys.path.append(os.path.join(SUMO_HOME, 'tools'))
import traci

# ── Constantes del agente ──
SEMAFORO_ID     = 'J1'
FASES_VERDE     = [0, 2]          # índices de fases en verde
DURACION_MIN    = 10              # segundos mínimos de verde
DURACION_MAX    = 60              # segundos máximos de verde
PASOS_SIMULACION = 1800           # duración total de un episodio (segundos)
ZONA_DEMO       = 'Av. 80'       # zona SIPOM que representa esta simulación


# ──────────────────────────────────────────────
# Entorno Gymnasium
# ──────────────────────────────────────────────

class SUMOSIPOMEnv(gym.Env):
    """
    Entorno de simulación de tráfico para el agente PPO.

    Observación (8 valores):
        - Vehículos esperando en cada carril (4 carriles)
        - Velocidad promedio de vehículos (normalizada)
        - Fase actual del semáforo
        - Nivel de congestión predicho por SIPOM (0/1/2)
        - Lluvia (0/1)

    Acción (discreta):
        - 0: mantener fase actual
        - 1: cambiar a siguiente fase
        - 2: extender verde 10 segundos
        - 3: reducir verde 10 segundos
    """

    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.array([100, 100, 100, 100, 1.0, 3.0, 2.0, 1.0], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(4)

        # Cargar modelo SIPOM para guiar las decisiones del agente
        self.modelo_sipom = cargar_modelo(RUTA_MODELO)

        self.paso          = 0
        self.duracion_fase = 30
        self.sumo_activo   = False

    def _iniciar_sumo(self):
        """Lanza SUMO con TraCI."""
        if self.sumo_activo:
            traci.close()

        cmd = [
            SUMO_BINARY,
            '-n', NETWORK,
            '-r', ROUTES,
            '--no-warnings',
            '--no-step-log',
            '--time-to-teleport', '-1',
        ]
        traci.start(cmd)
        self.sumo_activo = True

    def _obtener_observacion(self):
        """
        Lee el estado actual de SUMO y lo convierte en el vector de observación.
        """
        try:
            carriles = traci.trafficlight.getControlledLanes(SEMAFORO_ID)
            carriles = list(dict.fromkeys(carriles))[:4]   # máximo 4 únicos

            # Vehículos esperando por carril (velocidad < 0.1 m/s)
            espera = []
            for c in carriles:
                vehs = traci.lane.getLastStepHaltingNumber(c)
                espera.append(float(vehs))
            while len(espera) < 4:
                espera.append(0.0)

            # Velocidad promedio normalizada
            ids_vehs = traci.vehicle.getIDList()
            if ids_vehs:
                vel_prom = np.mean([traci.vehicle.getSpeed(v) for v in ids_vehs])
                vel_norm = min(vel_prom / 13.89, 1.0)   # normalizar sobre vel máxima
            else:
                vel_norm = 1.0

            # Fase actual del semáforo
            fase_actual = float(traci.trafficlight.getPhase(SEMAFORO_ID))

            # Predicción SIPOM para esta condición
            hora_actual = (self.paso // 3600) % 24
            resultado   = predecir(
                modelo=self.modelo_sipom,
                zona=ZONA_DEMO,
                hora=hora_actual,
                dia_semana=1,
                mes=6,
                temperatura=22.0,
                lluvia=0,
                velocidad_kmh=vel_norm * 50,
                volumen_vehiculos=int(sum(espera) * 20),
                incidentes_hora=0,
            )
            nivel_congestion = float(resultado['clase'])
            lluvia           = 0.0

            obs = np.array(
                espera[:4] + [vel_norm, fase_actual, nivel_congestion, lluvia],
                dtype=np.float32
            )

        except Exception:
            obs = np.zeros(8, dtype=np.float32)

        return obs

    def _calcular_recompensa(self, obs):
        """
        Recompensa = velocidad promedio - penalización por espera - penalización por congestión.
        El agente aprende a maximizar fluidez y minimizar tiempo de espera.
        """
        espera_total     = sum(obs[:4])
        vel_norm         = obs[4]
        nivel_congestion = obs[6]

        recompensa = (
            + vel_norm * 2.0              # premio por velocidad alta
            - espera_total * 0.05         # penalización por vehículos esperando
            - nivel_congestion * 0.5      # penalización por congestión predicha
        )
        return float(recompensa)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._iniciar_sumo()
        self.paso          = 0
        self.duracion_fase = 30
        obs = self._obtener_observacion()
        return obs, {}

    def step(self, accion):
        try:
            fase_actual = traci.trafficlight.getPhase(SEMAFORO_ID)

            if accion == 0:
                # Mantener fase actual — no hacer nada
                pass

            elif accion == 1:
                # Cambiar a la siguiente fase
                siguiente = (fase_actual + 1) % 4
                traci.trafficlight.setPhase(SEMAFORO_ID, siguiente)

            elif accion == 2:
                # Extender verde 10 segundos
                self.duracion_fase = min(self.duracion_fase + 10, DURACION_MAX)
                traci.trafficlight.setPhaseDuration(SEMAFORO_ID, self.duracion_fase)

            elif accion == 3:
                # Reducir verde 10 segundos
                self.duracion_fase = max(self.duracion_fase - 10, DURACION_MIN)
                traci.trafficlight.setPhaseDuration(SEMAFORO_ID, self.duracion_fase)

            # Avanzar simulación 5 pasos
            for _ in range(5):
                traci.simulationStep()
                self.paso += 1

        except Exception as e:
            print(f'Error en step: {e}')

        obs        = self._obtener_observacion()
        recompensa = self._calcular_recompensa(obs)
        terminado  = self.paso >= PASOS_SIMULACION
        truncado   = False

        return obs, recompensa, terminado, truncado, {}

    def close(self):
        if self.sumo_activo:
            try:
                traci.close()
            except Exception:
                pass
            self.sumo_activo = False


# ──────────────────────────────────────────────
# Entrenamiento del agente PPO
# ──────────────────────────────────────────────

def entrenar_agente(pasos_total: int = 50_000):
    """
    Entrena el agente PPO y guarda el modelo en sumo/sipom_ppo.zip
    """
    print('Verificando entorno...')
    env = SUMOSIPOMEnv()
    check_env(env, warn=True)

    print(f'Entrenando agente PPO — {pasos_total:,} pasos...')
    modelo_ppo = PPO(
        policy             = 'MlpPolicy',
        env                = env,
        learning_rate      = 3e-4,
        n_steps            = 512,
        batch_size         = 64,
        n_epochs           = 10,
        gamma              = 0.95,
        gae_lambda         = 0.95,
        clip_range         = 0.2,
        verbose            = 1,
    )
    modelo_ppo.learn(total_timesteps=pasos_total)

    ruta_out = os.path.join(os.path.dirname(__file__), 'sipom_ppo')
    modelo_ppo.save(ruta_out)
    print(f'✅ Agente guardado en: {ruta_out}.zip')

    env.close()
    return modelo_ppo


def demo_agente():
    """
    Corre una demo del agente entrenado durante 1 episodio completo
    e imprime estadísticas por cada 100 pasos.
    """
    ruta_ppo = os.path.join(os.path.dirname(__file__), 'sipom_ppo.zip')

    if not os.path.exists(ruta_ppo):
        print('❌ No se encontró sipom_ppo.zip — ejecuta entrenar_agente() primero.')
        return

    print('Cargando agente entrenado...')
    env        = SUMOSIPOMEnv()
    modelo_ppo = PPO.load(ruta_ppo, env=env)

    obs, _          = env.reset()
    recompensa_total = 0
    paso             = 0

    print('\nIniciando demo...\n')
    while True:
        accion, _ = modelo_ppo.predict(obs, deterministic=True)
        obs, rew, done, truncated, _ = env.step(int(accion))
        recompensa_total += rew
        paso             += 1

        if paso % 20 == 0:
            nivel = ['Fluido', 'Moderado', 'Crítico'][int(obs[6])]
            print(
                f'Paso {paso:4d} | '
                f'Acción: {accion} | '
                f'Espera: {sum(obs[:4]):.0f} veh | '
                f'Vel: {obs[4]*50:.1f} km/h | '
                f'Congestión: {nivel} | '
                f'Recompensa acum: {recompensa_total:.2f}'
            )

        if done or truncated:
            break

    print(f'\n✅ Demo finalizada — Recompensa total: {recompensa_total:.2f}')
    env.close()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == '__main__':
    # Paso 1: entrenar
    entrenar_agente(pasos_total=50_000)

    # Paso 2: demo
    demo_agente()