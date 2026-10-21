readme_content = """# Fake News Challenge - Dual-Branch BiLSTM Model

Este repositorio contiene la implementación de un modelo de Aprendizaje Profundo diseñado para resolver el problema de **Stance Detection** (Detección de Postura) propuesto en el **Fake News Challenge (FNC-1)**.

El objetivo es clasificar la relación entre el encabezado (*Headline*) de una noticia y el cuerpo del artículo (*Article Body*) en una de las siguientes cuatro categorías:
1. `agree` (De acuerdo)
2. `disagree` (En desacuerdo)
3. `discuss` (Discusión/Debate)
4. `unrelated` (No relacionados)

## 📋 Arquitectura del Modelo

El modelo utiliza una arquitectura de **doble rama** (Dual-Branch) procesada mediante Redes Neuronales Recurrentes Bi-direccionales (BiLSTM):

* **Rama del Encabezado (Head Branch):** Procesa el título de la noticia limitando la secuencia a un máximo de 15 palabras.
* **Rama del Cuerpo (Body Branch):** Procesa el texto de la noticia limitando la secuencia a un máximo de 300 palabras.
* **Capa de Embeddings Consistente:** Ambos canales alimentan sus secuencias a una capa de Embedding inicializada con vectores preentrenados de **GloVe (Twitter 27B - 100 dimensiones)**. La dimensión se mantiene unificada en `EMBEDDING_DIM = 100` de forma estricta.
* **Fusión y Clasificación:** Las salidas de ambas ramas BiLSTM se concatenan y pasan por capas densas totalmente conectadas (`Dense` de 100 y 50 neuronas respectivamente con activaciones *ReLU* y capas *Dropout* al 50% para mitigar el sobreajuste) antes de la capa final de salida con activación *Softmax* para las 4 clases.

## ⚙️ Hiperparámetros y Configuración

* **Tamaño del Vocabulario:** Limitado a las 25,000 palabras más frecuentes.
* **Dimensión de Embedding (`EMBEDDING_DIM`):** 100
* **Dimensión de LSTM (`LSTM_DIM`):** 100
* **Tamaño de Lote (Batch Size):** 128
* **Épocas Máximas:** 40 (con callback de parada temprana)
* **Optimizador:** Adam
* **Función de Pérdida:** Categorical Crossentropy

## 🚀 Entrenamiento y Resultados

* El modelo se entrenó usando el conjunto oficial de entrenamiento y validación provisto en el Fake News Challenge.
* Se obtuvo una precisión (accuracy) aproximada del **85%** en el conjunto de validación.
* Las predicciones finales sobre el conjunto de test se guardaron exitosamente alineando cada par de noticia con su respectiva etiqueta textual original (`unrelated`, `agree`, `disagree`, `discuss`).

"""