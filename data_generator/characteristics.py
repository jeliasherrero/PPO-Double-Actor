import numpy as np

def generar_matriz(num_ejemplos_por_clase=100, dimensiones=2, seed=42):
    np.random.seed(seed)
    datos = []
    etiquetas = []

    # Clase 0: Nube gaussiana centrada en (0, 0)
    patron0 = np.random.normal(loc=0, scale=1.0, size=(num_ejemplos_por_clase, dimensiones))
    datos.append(patron0)
    etiquetas.append(np.zeros(num_ejemplos_por_clase))

    # Clase 1: Nube gaussiana centrada en (5, 5)
    patron1 = np.random.normal(loc=5, scale=1.0, size=(num_ejemplos_por_clase, dimensiones))
    datos.append(patron1)
    etiquetas.append(np.ones(num_ejemplos_por_clase))

    # Unimos
    X = np.vstack(datos)
    y = np.hstack(etiquetas)
    return X, y