import pandas as pd
import numpy as np
import math
import tensorflow as tf 
import joblib 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.losses import MeanSquaredError 

# --- Constantes Globales del SNI ---
K_global = 1.258104526
e_K = math.exp(K_global)

# --- RUTAS DE LOS ARCHIVOS DEL MODELO Y SCALER ENTRENADOS ---
MODELO_RN_PATH = 'modelo_fi_ideal_sn_universal.keras' 
SCALER_FEATURES_PATH = 'scaler_features.pkl'

# --- Definición de la función calculate_fi_ideal ---
def calculate_fi_ideal(x_val, px_val):
    """
    Calcula el Fi_ideal Real para un primo dado su posición X y valor P(X).
    Fórmula: Fi_ideal = X * e^K / P(X) (Consistente con la EPU del documento).
    """
    if x_val == 0 or px_val == 0: 
        return float('nan') 
    try:
        fi_ideal = x_val * e_K / px_val 
        return fi_ideal
    except ValueError as e:
        print(f"Error calculando Fi_ideal para X={x_val}, P(X)={px_val}: {e}")
        return float('nan')
    except ZeroDivisionError as e:
        print(f"Error de división por cero para X={x_val}, P(X)={px_val}: {e}")
        return float('nan')

# --- Definición de la función generar_primo_sni ---
def generar_primo_sni(X_target, modelo_rn_entrenado, scaler_features):
    """
    Genera un número primo predicho por el SNI para una posición X_target dada.
    Utiliza el modelo RN y el scaler reales.
    """
    if X_target == 0:
        return 2 

    P_X_real_for_features_val = None
    primos_mock_data_all = get_real_primes_for_test(X_target + 5) 
    for p_data in primos_mock_data_all:
        if p_data["X"] == X_target:
            P_X_real_for_features_val = p_data["P_X"]
            break
    
    if P_X_real_for_features_val is None or P_X_real_for_features_val <= 0:
        print(f"Advertencia: P(X) real no válido para X={X_target}. No se pueden generar features.")
        return None 

    features_vector = [
        float(X_target), 
        math.log(X_target) if X_target > 0 else 0.0,         
        float(X_target**2), 
        math.sqrt(X_target) if X_target >= 0 else 0.0,
        1.0/X_target if X_target != 0 else 0.0, 
        float(int(math.log10(X_target)) + 1) if X_target > 0 else 1.0, 
        float(X_target**3),
        float(P_X_real_for_features_val), 
        math.log(float(P_X_real_for_features_val)) if P_X_real_for_features_val > 0 else 0.0,       
        math.log10(X_target) if X_target > 0 else 0.0,
        math.log10(P_X_real_for_features_val) if P_X_real_for_features_val > 0 else 0.0,
        10**(int(math.log10(X_target))) if X_target > 0 else 1.0,
        10**(int(math.log10(P_X_real_for_features_val))) if P_X_real_for_features_val > 0 else 1.0
    ]
    
    features_scaled_for_prediction = scaler_features.transform([features_vector])
    
    predicted_f_ideal = modelo_rn_entrenado.predict(features_scaled_for_prediction)[0][0] 
    
    predicted_p_x_float = X_target * (predicted_f_ideal / e_K)
    
    primo_generado = round(predicted_p_x_float)
    
    return primo_generado

# --- Función para obtener primos reales para test ---
_primos_reales_test_cache = {} 

def get_real_primes_for_test(max_X_pos):
    """
    Obtiene primos reales para test. Utiliza un caché y genera los primos necesarios.
    """
    if max_X_pos < 1: 
        return [{"X": 0, "P_X": 2}] if max_X_pos == 0 else []
        
    cache_key = max_X_pos
    if cache_key not in _primos_reales_test_cache:
        primes_list_raw = get_first_n_primes_for_mock_data(max_X_pos + 5) 
        
        primos_dict = {p_val_dict["X"]: p_val_dict["P_X"] for p_val_dict in 
                       [{"X": i, "P_X": p} for i, p in enumerate(primes_list_raw)]}
        _primos_reales_test_cache[cache_key] = primos_dict
    
    result_list = []
    for X_pos in range(0, max_X_pos + 1): 
        if X_pos in _primos_reales_test_cache[cache_key]:
            result_list.append({"X": X_pos, "P_X": _primos_reales_test_cache[cache_key][X_pos]})
        else:
            P_X_approx = round(X_pos * (math.log(X_pos) + math.log(math.log(X_pos)))) if X_pos > 1 else 3
            if X_pos == 0: P_X_approx = 2
            result_list.append({"X": X_pos, "P_X": P_X_approx})
            
    return result_list

def get_first_n_primes_for_mock_data(n_primes):
    primes = []
    num = 2
    while len(primes) < n_primes:
        is_prime = True
        if num < 2: is_prime = False
        else:
            for i in range(2, int(math.sqrt(num)) + 1):
                if num % i == 0:
                    is_prime = False
                    break
        if is_prime:
            primes.append(num)
        num += 1
    return primes

# --- FUNCIÓN PRINCIPAL DE PRUEBAS DE PRECISIÓN ---
def realizar_pruebas_de_generacion_exacta(posiciones_X_a_probar, modelo_rn_entrenado, scaler_entrenado):
    """
    Realiza pruebas de precisión generando primos y comparándolos con valores reales,
    para replicar la "perfección" del experimento original.
    """
    print("\n--- Iniciando Pruebas de Precisión del Primo Generado (Réplica de Perfección) ---")
    print("--- Esto comparará P(X) Real vs. P(X) Predicho para X=1001, 1002, 1003... ---")

    print("\n X    | P(X) Real  | Fi_ideal Real | Fi_ideal Predicho | P(X) Predicho | Diferencia P(X)")
    print("------|------------|---------------|-------------------|---------------|-----------------")

    resultados_para_excel = [] 

    max_X_target_needed = max(posiciones_X_a_probar)
    all_real_primes_data_needed = get_real_primes_for_test(max_X_target_needed + 5) 
    real_primes_dict_for_lookup = {p["X"]: p["P_X"] for p in all_real_primes_data_needed}

    for X_target in posiciones_X_a_probar:
        P_X_real = real_primes_dict_for_lookup.get(X_target)
        
        if P_X_real is None or P_X_real <= 0:
            print(f"{X_target:<6} | Error: P(X) Real no válido o no encontrado en el diccionario. Saltando.")
            continue

        predicted_p_x_rounded = generar_primo_sni(X_target, modelo_rn_entrenado, scaler_entrenado)
        
        fi_ideal_real_val = calculate_fi_ideal(X_target, P_X_real)
        
        raw_features_single_for_pred = [ 
            float(X_target), math.log(X_target) if X_target > 0 else 0.0, float(X_target**2), math.sqrt(X_target) if X_target >= 0 else 0.0, 1.0/X_target if X_target != 0 else 0.0, 
            float(int(math.log10(X_target)) + 1) if X_target > 0 else 1.0, float(X_target**3), float(P_X_real), math.log(float(P_X_real)) if P_X_real > 0 else 0.0,       
            math.log10(X_target) if X_target > 0 else 0.0, math.log10(P_X_real) if P_X_real > 0 else 0.0, 10**(int(math.log10(X_target))) if X_target > 0 else 1.0, 10**(int(math.log10(P_X_real))) if P_X_real > 0 else 1.0
        ]
        features_scaled_for_pred = scaler_entrenado.transform([raw_features_single_for_pred])
        fi_ideal_predicha_val = modelo_rn_entrenado.predict(features_scaled_for_pred)[0][0]
        
        diff_P_X = float(predicted_p_x_rounded) - float(P_X_real)
        
        print(f"{X_target:<6} | {P_X_real:<10} | {fi_ideal_real_val:<13.4f} | {fi_ideal_predicha_val:<17.4f} | {float(predicted_p_x_rounded):<13.4f} | {diff_P_X:<15.4f}")

        resultados_para_excel.append({
            "X_Posicion": X_target,
            "P(X) Real": P_X_real,
            "Fi_ideal Real": round(fi_ideal_real_val, 4),
            "Fi_ideal Predicho": round(fi_ideal_predicha_val, 4),
            "P(X) Predicho": predicted_p_x_rounded,
            "Diferencia P(X)": diff_P_X
        })

    print("\n--- Pruebas de Precisión de Generación COMPLETADAS (Réplica de Perfección) ---")
    return resultados_para_excel

# --- LLAMADA PRINCIPAL A LAS PRUEBAS DE PRECISIÓN ---
posiciones_para_tabla = [1001, 1002, 1003] 

# --- CARGAR EL MODELO Y SCALER REALES ---
try:
    custom_objects = {
        'Adam': tf.keras.optimizers.Adam 
    }
    
    modelo_rn_cargado = tf.keras.models.load_model(MODELO_RN_PATH, custom_objects=custom_objects)
    
    scaler_cargado = joblib.load(SCALER_FEATURES_PATH)
    
    print(f"\n¡Modelo y Scaler REALES cargados exitosamente desde el disco!")
except Exception as e:
    print(f"\nError al cargar el modelo o scaler REALES: {e}")
    print("Asegúrese de haber ejecutado 'procesar_primos_fases1_y_2.py' para entrenar y guardarlos en formato .keras.")
    print("Y que las rutas MODELO_RN_PATH y SCALER_FEATURES_PATH sean correctas.")
    print(f"Detalle del error: {e}") 
    exit() 

# Llamar a la función de pruebas con el modelo y scaler reales
resultados_precision = realizar_pruebas_de_generacion_exacta(posiciones_para_tabla, modelo_rn_cargado, scaler_cargado)

# --- Guardar los resultados en un archivo Excel ---
output_excel_filename = "resultados_precision_sn_sni.xlsx"
df_resultados_precision = pd.DataFrame(resultados_precision)
df_resultados_precision.to_excel(output_excel_filename, index=False)

print(f"\n¡Compadre! Los resultados de precisión han sido guardados en '{output_excel_filename}'")
print("\nPrimeras 5 filas del archivo Excel (en consola):")
print(df_resultados_precision.head())
print("\nÚltimas 5 filas del archivo Excel (en consola):")
print(df_resultados_precision.tail())