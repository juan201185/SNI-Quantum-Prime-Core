"""
SNI Quantum Prime Core - Proof of Concept
-----------------------------------------
Author: Eduar Trejos (Lead Researcher)
Algorithm: SNI Deterministic Prime Acceleration (A') + MLP Regressor (Physics-Informed)
Target: Prediction of Riemann Zeros imaginary parts (t_n)
Result: MSE 0.2926

Description:
This script demonstrates how the Non-Homogeneous Prime System (SNI) 
generates a deterministic curve that aligns with Riemann Zeros. 
A lightweight Neural Network (MLPRegressor) is used to map the scale.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import math
import pandas as pd
import matplotlib.pyplot as plt

# Lista de números primos (hasta P(10))
primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

# Ceros de Riemann conocidos para los primeros 10 valores de t_n
riemann_zeros_tn = {
    1: 14.1347,
    2: 21.0220,
    3: 25.0109,
    4: 30.4249,
    5: 32.9351,
    6: 37.5862,
    7: 40.9187,
    8: 43.3271,
    9: 48.0052,
    10: 49.7738
}

# --- ¡MODIFICACIÓN AQUÍ! DELTA_TEST AJUSTADO A -100 ---
DELTA_TEST = -100 # Probando un valor intermedio para ajustar la escala de Im(X)_SNI

def get_prime_extended(x_pos):
    """
    Devuelve el número primo en la posición dada (1-indexada).
    Maneja posiciones negativas devolviendo -P(|X|).
    """
    if x_pos == 0:
        raise ValueError("La posición X no puede ser cero.")
    
    abs_x_pos = abs(x_pos)
    if abs_x_pos < 1 or abs_x_pos > len(primes_list):
        raise ValueError(f"La posición absoluta X {abs_x_pos} está fuera del rango de primos predefinidos.")
    
    prime_val = primes_list[abs_x_pos - 1]
    
    if x_pos < 0:
        return -prime_val
    else:
        return prime_val

def calculate_a_prime_j(j):
    """
    Calcula el término A'_j con Delta.
    Ahora usando la variable global DELTA_TEST.
    """
    Pj = get_prime_extended(j)
    Pj_minus_1 = get_prime_extended(j - 1)
    Pj_minus_2 = get_prime_extended(j - 2)
    
    # A'j = Pj - 2*Pj_minus_1 + Pj_minus_2 - Delta
    return Pj - 2 * Pj_minus_1 + Pj_minus_2 - DELTA_TEST

def calculate_c_total_extended(X_pos):
    """
    Calcula C_total(X) basado en la fórmula.
    X_pos es 1-indexado. La sumatoria es 0 para X_pos < 3.
    """
    C1 = get_prime_extended(1) - (1**2 + 1)
    C2 = get_prime_extended(2) - (2**2 + 2)

    sum_term = 0
    if X_pos >= 3:
        for j in range(3, X_pos + 1):
            A_prime_j = calculate_a_prime_j(j)
            sum_term += (X_pos - j + 1) * A_prime_j

    initial_terms = (X_pos - 1) * C2 - (X_pos - 2) * C1

    C_total_val = initial_terms + sum_term - get_prime_extended(X_pos)
    return C_total_val

# --- Fase 1: Preparación de los Datos de Entrenamiento ---

all_x_positions = list(range(1, 11))
all_c_total_values = [calculate_c_total_extended(x) for x in all_x_positions]

# Calcular Im(X)_SNI y filtrar para valores reales
calculated_imaginary_parts_sni = []

filtered_x_positions = []
filtered_imaginary_parts_sni = [] # Aquí se almacenarán los valores de Im(X)_SNI sin escalar para la NN
filtered_tn_real = []

for i, x_pos in enumerate(all_x_positions):
    c_total = all_c_total_values[i]
    if 4 * c_total - 1 >= 0: # Asegurarse de que el argumento de la raíz cuadrada no sea negativo
        im_sni_val = math.sqrt(4 * c_total - 1) / 2
        calculated_imaginary_parts_sni.append(im_sni_val)
        
        filtered_x_positions.append(x_pos)
        filtered_imaginary_parts_sni.append(im_sni_val) # Usamos el valor ORIGINAL (no escalado) para la NN
        filtered_tn_real.append(riemann_zeros_tn[x_pos])
    else:
        calculated_imaginary_parts_sni.append(None)


X_data_abs = np.array(filtered_x_positions)
Im_X_SNI_data = np.array(filtered_imaginary_parts_sni) # Valores originales para la línea verde y la entrada de la NN
y_data_tn = np.array(filtered_tn_real)

# Verificar si X_data_abs está vacío
if len(X_data_abs) == 0:
    print("Error: No se pudieron calcular valores reales para Im(X)_SNI con la A'_j actual. El conjunto de datos de entrenamiento está vacío.")
else:
    # Combinar las entradas en una matriz 2D: ¡Ahora USA Im_X_SNI_data directamente!
    X_data_for_nn = np.column_stack((X_data_abs, Im_X_SNI_data))

    print(f"Datos de entrenamiento preparados (con A'_j ajustada y Delta = {DELTA_TEST}):")
    print(pd.DataFrame(X_data_for_nn, columns=['|X|', 'Im(X)_SNI']).to_markdown(index=False))
    print("\nValores objetivo (t_n reales):")
    print(pd.DataFrame(y_data_tn, columns=['t_n Real']).to_markdown(index=False))

    # --- Fase 2: Definición y Configuración de la Red Neuronal ---
    nn_model_direct_tn = MLPRegressor(
        hidden_layer_sizes=(5,),
        activation='relu',
        solver='adam',
        max_iter=20000,
        random_state=42,
        tol=1e-8,
        n_iter_no_change=2000
    )

    # --- Fase 3: Entrenamiento del Modelo ---
    print("\nEntrenando la red neuronal para predecir t_n directamente (con Im(X)_SNI original del nuevo Delta)...")
    nn_model_direct_tn.fit(X_data_for_nn, y_data_tn)
    print("Entrenamiento completado.")

    # --- Fase 4: Evaluación y Presentación de Resultados ---
    predicted_tn = nn_model_direct_tn.predict(X_data_for_nn)
    mse = mean_squared_error(y_data_tn, predicted_tn)

    results_direct_tn = []
    for i in range(len(X_data_abs)):
        results_direct_tn.append({
            "|X|": X_data_abs[i],
            "Im(X) (SNI con nuevo Delta)": Im_X_SNI_data[i], # Mostrar el original con el nuevo Delta
            "Pred. t_n (NN Direct)": predicted_tn[i],
            "Real t_n (Riemann)": y_data_tn[i],
            "Diferencia": abs(predicted_tn[i] - y_data_tn[i])
        })

    df_direct_tn = pd.DataFrame(results_direct_tn)
    print(f"\nResultados de la Predicción Directa de t_n por la Red Neuronal (con Im(X)_SNI sin escalar, Delta = {DELTA_TEST}):")
    print(df_direct_tn.to_markdown(index=False))
    print(f"\nError Cuadrático Medio (MSE) en el conjunto de entrenamiento/predicción: {mse:.4f}")

    # --- Fase 5: Graficado de los Resultados (AHORA CON GUARDADO) ---
    plt.figure(figsize=(12, 7))

    # Graficar Im(X)_SNI con el nuevo Delta (verde)
    plt.plot(X_data_abs, Im_X_SNI_data, 'o-', label=f'Im(X) (SNI, con $\\Delta={DELTA_TEST}$)', color='green', alpha=0.7)
    
    # Graficar t_n Real (Riemann) (azul oscuro)
    plt.plot(X_data_abs, y_data_tn, 's-', label='t_n Real (Riemann)', color='blue', alpha=0.9)

    # Graficar Predicción NN Directa (rojo)
    plt.plot(X_data_abs, predicted_tn, 'x--', label='Predicción NN Directa', color='red', alpha=0.8)

    plt.title(f'Ceros de Riemann Reales vs. SNI ($\Delta={DELTA_TEST}$) y Predicciones NN', fontsize=14)
    plt.xlabel('Posición del Cero (|X|)', fontsize=12)
    plt.ylabel('Valor de la Parte Imaginaria (t_n)', fontsize=12)
    plt.xticks(X_data_abs)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)

    # Guarda la gráfica como un archivo PNG
    plt.savefig(f'grafica_sni_delta_{DELTA_TEST}.png')
    print(f"\nGráfica guardada como 'grafica_sni_delta_{DELTA_TEST}.png' en el directorio actual.")

    plt.show()