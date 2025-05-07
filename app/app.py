# app/app.py
import streamlit as st
import joblib
import numpy as np
import os

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Predictor de Especies de Iris",
    page_icon="🌸",
    layout="centered"
)

# --- Carga del Modelo ---
# Construir la ruta al modelo de forma robusta
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'iris_model.pkl')

@st.cache_resource # Usar st.cache_resource para modelos y conexiones
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Archivo del modelo no encontrado en {model_path}. "
                 "Asegúrate de que el modelo haya sido entrenado y guardado.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_model(MODEL_PATH)

# Clases de Iris (para mostrar el resultado)
iris_species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# --- Interfaz de Usuario ---
st.title("🌸 Predictor de Especies de Iris")
st.markdown("""
Esta aplicación utiliza un modelo de Machine Learning para predecir la especie de una flor Iris
basándose en las medidas de sus sépalos y pétalos.
""")

if model is not None:
    st.sidebar.header("Parámetros de Entrada:")

    def user_input_features():
        sepal_length = st.sidebar.slider('Longitud del Sépalo (cm)', 4.0, 8.0, 5.4, 0.1)
        sepal_width = st.sidebar.slider('Ancho del Sépalo (cm)', 2.0, 4.5, 3.4, 0.1)
        petal_length = st.sidebar.slider('Longitud del Pétalo (cm)', 1.0, 7.0, 1.3, 0.1)
        petal_width = st.sidebar.slider('Ancho del Pétalo (cm)', 0.1, 2.5, 0.2, 0.1)
        data = {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
        features = np.array([data['sepal_length'], data['sepal_width'],
                             data['petal_length'], data['petal_width']]).reshape(1, -1)
        return features, data

    input_features, input_data_display = user_input_features()

    st.subheader("Valores de Entrada Seleccionados:")
    st.write(input_data_display)

    if st.button('Predecir Especie'):
        prediction = model.predict(input_features)
        prediction_proba = model.predict_proba(input_features)

        st.subheader('Predicción:')
        predicted_specie_name = iris_species[prediction[0]]
        st.success(f"La especie predicha es: **{predicted_specie_name}**")

        st.subheader('Probabilidades de Predicción:')
        # Crear un diccionario para mostrar probabilidades con nombres de clase
        prob_dict = {iris_species[i]: prob for i, prob in enumerate(prediction_proba[0])}
        st.write(prob_dict)
else:
    st.warning("El modelo no está disponible. No se pueden realizar predicciones.")

st.markdown("---")
st.markdown("### Documentación del Modelo")
st.markdown("""
**Variables de Entrada:**
*   **Longitud del Sépalo (cm):** Tipo `float`. Rango esperado: 4.0 - 8.0 cm.
*   **Ancho del Sépalo (cm):** Tipo `float`. Rango esperado: 2.0 - 4.5 cm.
*   **Longitud del Pétalo (cm):** Tipo `float`. Rango esperado: 1.0 - 7.0 cm.
*   **Ancho del Pétalo (cm):** Tipo `float`. Rango esperado: 0.1 - 2.5 cm.

**Salida del Modelo:**
*   **Especie Predicha:** Tipo `string`. Valores posibles: 'Setosa', 'Versicolor', 'Virginica'.
*   **Probabilidades de Predicción:** Un diccionario o array con las probabilidades para cada una de las tres especies.
""")