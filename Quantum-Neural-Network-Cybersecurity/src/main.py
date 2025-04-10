from src.utils import generate_data
from src.classical_model import train_classical_model
from src.quantum_model import train_quantum_model

def main():
    print("🔐 Generando datos de amenazas cibernéticas...")
    df = generate_data()
    X, y = df.drop('label', axis=1), df['label']

    print("\n⚙️ Entrenando modelo clásico...")
    train_classical_model(X, y)

    print("\n⚛️ Entrenando modelo cuántico...")
    train_quantum_model(X, y)

    print("\n✅ Proyecto finalizado. Comparativa entre modelos completada.")

if __name__ == "__main__":
    main()
