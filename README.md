# Neural Prime Approximator (SNI Project)

## Overview
This project implements a **Deep Neural Network (DNN)** using **TensorFlow/Keras** to model deterministic behaviors in the distribution of prime numbers. Based on the **"Sistema Numérico Impecable" (SNI)** theoretical framework, the model learns to approximate the ideal contour function ($F_{ideal}$) to predict the location of prime numbers.

## Key Features
* **Architecture:** Multi-layer Perceptron (MLP) with `HeNormal` initialization.
* **Optimization:** Uses **Batch Normalization** for stability and the **Adam** optimizer with custom learning rates.
* **Feature Engineering:** Advanced input transformation (logarithmic, polynomial, and reciprocal scaling) to handle non-linear numerical relationships.
* **Data Pipeline:** Automated consolidation of large datasets (millions of primes) and `StandardScaler` normalization.

## Technical Stack
* **Language:** Python 3.x
* **Libraries:** TensorFlow, Keras, NumPy, Scikit-Learn, Joblib.

## Usage
The script `procesar_primos_fases1_y_2.py` handles the full pipeline:
1.  Data ingestion from raw text files.
2.  Calculation of the $F_{ideal}$ target variable.
3.  Training of the Neural Network.
4.  Model serialization (.keras format).

---
*Developed by a Researcher with a background in Physics and Computational Science.*
