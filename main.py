import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. CARGA DE DATOS
columns = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=columns)

# Filtrar sensores constantes
var_sensors = [col for col in df.columns if df[col].nunique() > 1 and col.startswith('s_')]

# 2. CÁLCULO DE PESOS DE MONOTONICIDAD (w_i)
def compute_sensor_weights(data, sensors):
    monotonicity = {}
    for s in sensors:
        corrs = []
        for unit, group in data.groupby('unit_nr'):
            if len(group) > 10:
                corr, _ = spearmanr(group['time_cycles'], group[s])
                if not np.isnan(corr):
                    corrs.append(abs(corr))
        monotonicity[s] = np.mean(corrs) if corrs else 0.0
    
    total_mon = sum(monotonicity.values())
    weights = {s: monotonicity[s] / total_mon for s in sensors}
    return weights

weights = compute_sensor_weights(df, var_sensors)

# 3. CÁLCULO DE Fi_ideal UNITARIO (POR MOTOR)
# Primeros N0 = 10 ciclos de cada motor
N0 = 10
nominal_df = df[df['time_cycles'] <= N0]

# Línea base por motor
Fi_ideal_unit = nominal_df.groupby('unit_nr')[var_sensors].mean().reset_index()
Fi_ideal_unit.columns = ['unit_nr'] + [f'{s}_ideal' for s in var_sensors]

# Desviación estándar nominal de la población para normalizar
sigma_nominal = nominal_df[var_sensors].std().replace(0, 1e-6)

# Fusionar la línea base ideal con el dataframe principal
df = df.merge(Fi_ideal_unit, on='unit_nr', how='left')

# 4. FORMULACIÓN DE D_u(t) Y P(X)
weighted_sq_diffs = pd.DataFrame(index=df.index)

for s in var_sensors:
    # Desviación estandarizada respecto a SU PROPIA línea base
    d_iu = (df[s] - df[f'{s}_ideal']) / sigma_nominal[s]
    # Aportación ponderada al cuadrado
    weighted_sq_diffs[s] = weights[s] * (d_iu ** 2)

# Distancia Mahalanobis/Euclídea ponderada
df['D_u'] = np.sqrt(weighted_sq_diffs.sum(axis=1))

# Factor de escala gamma para suavizar decaimiento
gamma = 0.35
df['P_X'] = np.exp(-gamma * df['D_u'])

# 5. PREPARACIÓN DE OBJETIVO RUL PIECEWISE
max_cycles = df.groupby('unit_nr')['time_cycles'].max()
df['RUL_raw'] = df['unit_nr'].map(max_cycles) - df['time_cycles']

# Acotación física: RUL_max = 125
RUL_MAX = 125
df['RUL_target'] = df['RUL_raw'].clip(upper=RUL_MAX)

# 6. ENTRENAMIENTO Y EVALUACIÓN
features = ['P_X', 'D_u', 'time_cycles']
X = df[features]
y = df['RUL_target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("--- RESULTADOS MODELO MATEMÁTICO PERFECCIONADO ---")
print(f"R² Score: {r2:.4f}")
print(f"RMSE:     {rmse:.4f} ciclos")

print("\nPesos de Monotonicidad asignados a Top 5 Sensores:")
sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
for sensor, w in sorted_w[:5]:
    print(f"  {sensor}: {w:.4f}")