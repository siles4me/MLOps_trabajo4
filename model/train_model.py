# model/train_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib
import os

# Crear directorio 'model' si no existe (para GitHub Actions)
os.makedirs(os.path.dirname(__file__), exist_ok=True)

def train_and_save_model():
    """
    Entrena un modelo simple de clasificación Iris y lo guarda.
    Reemplaza esto con tu lógica de entrenamiento.
    """
    print("Cargando datos Iris...")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Entrenando modelo Logistic Regression...")
    model = LogisticRegression(max_iter=200) # Aumentar max_iter para convergencia
    model.fit(X_train, y_train)

    model_filename = os.path.join(os.path.dirname(__file__), 'iris_model.pkl')
    print(f"Guardando modelo en {model_filename}...")
    joblib.dump(model, model_filename)
    print("Modelo entrenado y guardado exitosamente.")

    # (Opcional) Evaluar el modelo
    accuracy = model.score(X_test, y_test)
    print(f"Precisión del modelo en datos de prueba: {accuracy:.4f}")

if __name__ == "__main__":
    train_and_save_model()