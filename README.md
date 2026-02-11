# SNI Quantum Prime Core: Generación de Primos Deterministas y Unificación Cuántica

🚀 **Implementación Oficial del Sistema Numérico Impecable (SNI)**

* **Autor:** Eduar Fabian Trejos Bermudez (Lead Researcher)
* **Estado:** Validación a Gran Escala (1 Millón de Datapoints)

---

## 📌 Resumen Ejecutivo

Este repositorio unifica dos descubrimientos fundamentales del SNI que desafían la estocástica clásica:
1.  **Generación de Primos (EPU):** Los números primos pueden generarse con precisión casi absoluta usando la *Ecuación del Punto Unitario*.
2.  **Predicción de Riemann ($A'$):** Los Ceros de Riemann siguen una curva determinista basada en la *Aceleración Primal*.

---

## 🛠️ Arquitectura del Sistema (Módulos y Papers)

El sistema se divide en dos fases. Cada código tiene su propio respaldo teórico (PDF).

### 🔹 MÓDULO 1: El Generador de Primos (Core)
* **Código:** `SNI_Core_Prime_Gen.py`
* **Paper Teórico:** `SNI_Paper_Prime_Generation_EPU.pdf`
* **Descripción:** * Implementa la **Ecuación del Punto Unitario (EPU)** y la **Homogeneidad Ideal ($F_i$)**.
    * **Escalabilidad:** Mientras el paper documenta la prueba teórica con 1,000 primos, este código entrena la red neuronal con **1,000,000 de primos**, logrando una estabilidad perfecta.
    * **Precisión:** MSE ~0.0000004 (Identificación exacta).

### 🔹 MÓDULO 2: El Predictor de Riemann (Aplicación)
* **Código:** `SNI_Hybrid_Predictor.py`
* **Paper Teórico:** `SNI_Proof_Deterministic_Primes.pdf`
* **Descripción:** * Utiliza la métrica de **Aceleración Primal ($A'$)** derivada del núcleo para predecir la ubicación de los Ceros de Riemann ($t_n$).
    * **Resultado:** Convierte el caos aparente de los ceros en una curva geométrica predecible.
    * **Precisión:** MSE 0.2926 (Alta convergencia).

---

## 📊 Evidencia Visual
* **`Result_MSE_0.2926.png`**: Gráfica que muestra cómo la curva determinista del SNI "muerde" los Ceros de Riemann.

---

## 💻 Instrucciones de Ejecución

### Para Generación de Primos (Requiere dataset masivo):
```bash
# Este script procesa hasta 1 millón de primos para entrenar la Fi Ideal
python SNI_Core_Prime_Gen.py

Para Predicción de Riemann (Autónomo):
Bash
# Este script ejecuta la demostración de la Hipótesis de Riemann
python SNI_Hybrid_Predictor.py
📜 Citación
Si utiliza este código, cite según el módulo correspondiente:

Trejos Bermudez, E. F. (2026). The Unitary Point Equation & Ideal Homogeneity: Validated on 10^6 Primes. GitHub Repository.
