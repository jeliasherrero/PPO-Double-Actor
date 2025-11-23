from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Ruta raíz donde están los directorios MJ, R1, R2, R3, R4, etc.
ROOT_DIR = "MV"  # Cambia esto por la ruta correcta

# Subdirectorios que quieres buscar dentro de cada R
SUBDIRS = ["INT", "NEG", "POS"]

def juntar_datos(datos):
    Xt_list = []
    Yt_list = []
    for filename, X, Y in datos:
        Xt_list.append(X.to_numpy())  # Convertir DataFrame a numpy array
        Yt_list.append(Y.to_numpy())
    # Concatenar en un solo array
    Xt = np.vstack(Xt_list)
    Yt = np.vstack(Yt_list)
    return Xt, Yt

def leer_csv_en_subdirs(root_dir, subdirs):
    """
    Recorre los directorios R1, R2, R3, R4 dentro de root_dir,
    y en cada uno busca las subcarpetas en subdirs.
    Lee todos los CSV de esas subcarpetas y extrae X e Y.
    Devuelve una lista de tuplas (ruta_csv, X, Y).
    """
    resultados = []

    dirs = os.listdir(ROOT_DIR)
    # Recorrer los directorios R1, R2, R3, R4
    for r_dir in dirs:
        r_path = os.path.join(root_dir, r_dir)
        if not os.path.exists(r_path):
            continue

        for subdir in subdirs:
            subdir_path = os.path.join(r_path, subdir)
            if not os.path.exists(subdir_path):
                continue

            for fname in os.listdir(subdir_path):
                if not fname.lower().endswith(".csv"):
                    continue

                csv_path = os.path.join(subdir_path, fname)
                df = pd.read_csv(csv_path)

                # X: primeras columnas (por ejemplo, 0 a 9)
                # Y: columnas 11 y 12 (índices 10 y 11)
                X = df.iloc[:, [0,1,5,6]]      # primeras columnas
                Y = df.iloc[:, 10:12]    # columnas 11 y 12

                resultados.append((csv_path, X, Y))

    return resultados


result = leer_csv_en_subdirs(ROOT_DIR, SUBDIRS)
X, y = juntar_datos(result)
y = np.argmax(y, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Tasa de acierto: {accuracy}")

# Análisis PCA
X, y = juntar_datos(result)
y = np.argmax(y, axis=1)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)   # (20000, 2)

# 3. Representamos las dos primeras componentes, coloreando por clase
plt.figure(figsize=(6, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=5)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA (2 primeras componentes)')
plt.colorbar(scatter, label='Clase')
plt.tight_layout()
plt.show()