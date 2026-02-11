# SNI Quantum Prime Core: Generación de Primos Deterministas y Unificación Cuántica

🚀 **Implementación Oficial del Sistema Numérico Impecable (SNI)**

* **Autor:** Eduar Fabián Trejos Bermúdez (Lead Researcher)
* **Estado:** Validación Industrial (1 Millón de Datapoints)

---

## 📌 Resumen Ejecutivo

Este repositorio unifica dos descubrimientos fundamentales del SNI que desafían la estocástica clásica:

1.  **Generación de Primos (EPU):** Los números primos pueden generarse con precisión casi absoluta usando la **Ecuación del Punto Unitario**.
2.  **Predicción de Riemann ($A'$):** Los Ceros de Riemann siguen una curva determinista basada en la **Aceleración Primal**.

---

## 🛠️ Arquitectura del Sistema (Workflow)

El sistema opera en un flujo de **Entrenamiento → Validación → Aplicación**.

| Módulo | Archivo | Función |
| :--- | :--- | :--- |
| **1. Core (Entrenamiento)** | `SNI_Core_Prime_Gen.py` | **El Motor.** Procesa datasets masivos (1M primos) y entrena la red neuronal (SNI-Net) para aprender la *Homogeneidad Ideal ($F_i$)*. Genera el archivo del modelo `.keras`. |
| **2. Core (Validación)** | `SNI_Validator_Precision_Test.py` | **El Juez.** Carga el modelo entrenado y realiza pruebas de "fuego real", generando primos en posiciones específicas y calculando la precisión exacta (MSE ~0.0). |
| **3. App (Riemann)** | `SNI_Hybrid_Predictor.py` | **La Aplicación.** Utiliza la métrica derivada de *Aceleración Primal ($A'$)* para predecir la ubicación de los Ceros de Riemann ($t_n$) con un MSE de 0.2926. |

---

## 📂 Documentación Científica (Papers)

Cada módulo de código está respaldado por su respectiva demostración matemática:

* **`SNI_Paper_Prime_Generation_EPU.pdf`**: Fundamento teórico de la Ecuación del Punto Unitario (Base del Módulo 1 y 2).
* **`SNI_Proof_Deterministic_Primes.pdf`**: Demostración de la conexión entre el SNI y la Hipótesis de Riemann.

---

## 📊 Evidencia Visual

* **`Result_MSE_0.2926.png`**: Gráfica que muestra cómo la curva determinista del SNI "muerde" los Ceros de Riemann, eliminando la incertidumbre.

---

## 💻 Instrucciones de Ejecución

Para replicar los resultados, siga este orden lógico:

### Paso 1: Entrenar el Modelo (Generación de Primos)
Procesa los números primos y entrena la IA para entender la geometría SNI.
```bash
python SNI_Core_Prime_Gen.py
(Nota: Esto generará el archivo modelo_fi_ideal_sn_universal.keras)

Paso 2: Validar la Precisión
Prueba la exactitud del modelo generado en el paso anterior.

Bash
python SNI_Validator_Precision_Test.py
Paso 3: Ejecutar la Predicción de Riemann
Corre la simulación independiente para los Ceros de la Función Zeta.

Bash
python SNI_Hybrid_Predictor.py
📜 Citación
Si utiliza este código o teoría en su investigación, por favor cite:

Trejos Bermudez, E. F. (2026). The Unitary Point Equation & Ideal Homogeneity: Validated on 10^6 Primes. GitHub Repository.
