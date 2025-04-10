from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit.algorithms.optimizers import COBYLA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np

def train_quantum_model(X, y):
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_map = ZZFeatureMap(feature_dimension=4, reps=2)
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    # Simulación usando SVM cuántico
    from qiskit_machine_learning.algorithms import QSVC
    model = QSVC(quantum_kernel=kernel)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"[⚛️] Quantum Model Accuracy: {acc:.2f}")
    return model
