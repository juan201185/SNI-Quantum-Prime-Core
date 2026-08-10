import os
import urllib.request
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings
from scipy.stats import ConstantInputWarning

warnings.filterwarnings('ignore', category=ConstantInputWarning)

BASE_URL = "https://raw.githubusercontent.com/nkpro-data/CMAPSS-Data/main/"
SUBDATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
COLUMNS = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]

N0 = 10
RUL_MAX = 125
GAMMA = 0.35
EMA_SPAN = 15  # Ventana cinemática

def load_and_tag_data(sub_id, mode='train'):
    file_name = f'{mode}_{sub_id}.txt'
    if not os.path.exists(file_name):
        urllib.request.urlretrieve(BASE_URL + file_name, file_name)
    df = pd.read_csv(file_name, sep=r'\s+', header=None, names=COLUMNS)
    df['global_unit'] = sub_id + '_' + df['unit_nr'].astype(str)
    return df

print("1. Cargando y fusionando flota global (FD001-FD004)...")
df_train_all = pd.concat([load_and_tag_data(s, 'train') for s in SUBDATASETS], ignore_index=True)
df_test_all = pd.concat([load_and_tag_data(s, 'test') for s in SUBDATASETS], ignore_index=True)

rul_list = []
for s in SUBDATASETS:
    rul_file = f'RUL_{s}.txt'
    if not os.path.exists(rul_file):
        urllib.request.urlretrieve(BASE_URL + rul_file, rul_file)
    r = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['RUL_final'])
    r['global_unit'] = s + '_' + (r.index + 1).astype(str)
    rul_list.append(r)
rul_true_all = pd.concat(rul_list, ignore_index=True)

var_sensors = [col for col in df_train_all.columns if df_train_all[col].nunique() > 1 and col.startswith('s_')]
settings = ['setting_1', 'setting_2', 'setting_3']

print("2. Desacople de Régimen Global (K-Means K=6)...")
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df_train_all['regime'] = kmeans.fit_predict(df_train_all[settings])
df_test_all['regime'] = kmeans.predict(df_test_all[settings])

regime_means = df_train_all.groupby('regime')[var_sensors].mean()
regime_stds = df_train_all.groupby('regime')[var_sensors].std().replace(0, 1e-6)

for s in var_sensors:
    df_train_all[f'{s}_norm'] = (df_train_all[s] - df_train_all['regime'].map(regime_means[s])) / df_train_all['regime'].map(regime_stds[s])
    df_test_all[f'{s}_norm'] = (df_test_all[s] - df_test_all['regime'].map(regime_means[s])) / df_test_all['regime'].map(regime_stds[s])

norm_sensors = [f'{s}_norm' for s in var_sensors]

print("3. Monotonicidad Espectral (w_i)...")
monotonicity = {}
for s in norm_sensors:
    corrs = []
    for unit, group in df_train_all.groupby('global_unit'):
        if len(group) > 10:
            corr, _ = spearmanr(group['time_cycles'], group[s])
            if not np.isnan(corr):
                corrs.append(abs(corr))
    monotonicity[s] = np.mean(corrs) if corrs else 0.0

total_mon = sum(monotonicity.values())
weights = {s: monotonicity[s] / (total_mon + 1e-9) for s in norm_sensors}

print("4. Extrayendo Líneas Base y Proyectando D_u y P_X...")
train_nominal = df_train_all[df_train_all['time_cycles'] <= N0]
Fi_train = train_nominal.groupby('global_unit')[norm_sensors].mean().reset_index()
Fi_train.columns = ['global_unit'] + [f'{s}_ideal' for s in norm_sensors]
sigma_nominal = train_nominal[norm_sensors].std().replace(0, 1e-6)

df_train_all = df_train_all.merge(Fi_train, on='global_unit', how='left')
diffs_train = pd.DataFrame(index=df_train_all.index)
for s in norm_sensors:
    d_iu = (df_train_all[s] - df_train_all[f'{s}_ideal']) / sigma_nominal[s]
    diffs_train[s] = weights[s] * (d_iu ** 2)

df_train_all['D_u'] = np.sqrt(diffs_train.sum(axis=1))
df_train_all['P_X'] = np.exp(-GAMMA * df_train_all['D_u'])

print("5. Aplicando Filtro Cinemático (Memoria Estocástica)...")
def apply_kinematics(df):
    df['D_u_ema'] = df.groupby('global_unit')['D_u'].transform(lambda x: x.ewm(span=EMA_SPAN, adjust=False).mean())
    df['P_X_ema'] = df.groupby('global_unit')['P_X'].transform(lambda x: x.ewm(span=EMA_SPAN, adjust=False).mean())
    df['D_u_velocity'] = df.groupby('global_unit')['D_u_ema'].diff().fillna(0)
    return df

df_train_all = apply_kinematics(df_train_all)

max_cycles_train = df_train_all.groupby('global_unit')['time_cycles'].max()
df_train_all['RUL_raw'] = df_train_all['global_unit'].map(max_cycles_train) - df_train_all['time_cycles']
df_train_all['RUL_target'] = df_train_all['RUL_raw'].clip(upper=RUL_MAX)

print("6. Entrenando Arquitectura SNI + XGBoost...")
features = ['P_X', 'D_u', 'P_X_ema', 'D_u_ema', 'D_u_velocity', 'time_cycles'] + settings
X_train = df_train_all[features]
y_train = df_train_all['RUL_target']

# XGBoost calibrado para alta dimensionalidad ruidosa
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

print("7. Evaluando Prueba Ciega Oficial...")
test_nominal = df_test_all[df_test_all['time_cycles'] <= N0]
Fi_test = test_nominal.groupby('global_unit')[norm_sensors].mean().reset_index()
Fi_test.columns = ['global_unit'] + [f'{s}_ideal' for s in norm_sensors]

df_test_all = df_test_all.merge(Fi_test, on='global_unit', how='left')
diffs_test = pd.DataFrame(index=df_test_all.index)
for s in norm_sensors:
    d_iu = (df_test_all[s] - df_test_all[f'{s}_ideal']) / sigma_nominal[s]
    diffs_test[s] = weights[s] * (d_iu ** 2)

df_test_all['D_u'] = np.sqrt(diffs_test.sum(axis=1))
df_test_all['P_X'] = np.exp(-GAMMA * df_test_all['D_u'])

df_test_all = apply_kinematics(df_test_all)

last_cycles = df_test_all.groupby('global_unit').last().reset_index()
last_cycles = last_cycles.merge(rul_true_all, on='global_unit')

X_test_official = last_cycles[features]
y_test_official = last_cycles['RUL_final'].clip(upper=RUL_MAX)

y_pred_official = model.predict(X_test_official)

rmse = np.sqrt(mean_squared_error(y_test_official, y_pred_official))
r2 = r2_score(y_test_official, y_pred_official)

print("\n=========================================================")
print("   RESULTADOS SOTA: SNI + CINEMÁTICA + XGBOOST           ")
print("=========================================================")
print(f"Total de motores evaluados: {len(last_cycles)}")
print(f"R² Score Global: {r2:.4f}")
print(f"RMSE Global:     {rmse:.4f} ciclos")
print("=========================================================")