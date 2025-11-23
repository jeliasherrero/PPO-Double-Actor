import os
import sys
import pandas as pd
import numpy as np

from arguments import get_args
import torch
from torch.utils.data import TensorDataset, DataLoader

import gymnasium as gym
from network import FeedForwardNN, OrderedOutputNN, DecisionNN, RedOrdenada
from circePPO import PPO



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


def train(env,
          hyperparameters,
          actor_model,
          critic_model,
          decision_model,
          class_dim):
    print(f"Training...", flush=True)

    model = PPO(policy_class=RedOrdenada,
                critic_class=FeedForwardNN,
                decision_class=DecisionNN,
                class_dim=class_dim,
                env=env,
                **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '' and decision_model != '':
        print(f"Loading in {actor_model}, {critic_model} and {decision_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        model.decision_actor.load_state_dict(torch.load(decision_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '' or decision_model != '':
        # Don't train from scratch
        # if user accidentally forgets actors/critic model
        print(
            f"Error: Either specify all actor/critic/decision models or none at all. "
            f"We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    model.learn(total_timesteps=200_000_000)


def test(env, actor_model, class_dim=2):
    print(f"Testing {actor_model}...", flush=True)

    if actor_model == '':
        print(f"No actor model specified.", flush=True)
        sys.exit(0)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = FeedForwardNN(obs_dim,action_dim)
    policy.load_state_dict(torch.load(actor_model))

    eval_policy(policy, env)


def main(args=None):
    hyperparameters = {
        'batch_size': 64,
        'timesteps_per_batch': 1,
        'max_timesteps_per_episode': 64,
        'gamma': 0.99,
        'n_updates_per_iteration': 5,
        'lr': 3e-4,
        'clip': 0.2,
        'render': False,
        'render_every_i': 10,
        'save_freq': 10,
    }
    result = leer_csv_en_subdirs(ROOT_DIR, SUBDIRS)
    Xt, Yt = juntar_datos(result)

    Xt_tensor = torch.from_numpy(Xt).float()
    Yt_tensor = torch.from_numpy(Yt).float()
    dataset = TensorDataset(Xt_tensor, Yt_tensor)
    dataloader = DataLoader(dataset, batch_size=hyperparameters['batch_size'], shuffle=True, drop_last=True)
    # Asignamos drop_last = True para evitar el último batch si tiene distinto tamaño. Esto ocurre cuando
    # el tamaño del lote no es múltiplo del tamaño total de los datos
    # Lo hacemos así porque no vamos a recorrer el batch con un bucle for. Lo haremos con iter/next

    gym.register(
        id="gymnasium_env/circeENV-v0",
        entry_point="circeENV:TheusEnv",
        max_episode_steps=300,
    )
    env = gym.make('gymnasium_env/circeENV-v0',
                   batch_charact=dataloader,
                   n_charact=len(dataloader.dataset[0][0]))

    if args.mode == 'train':
        train(env,
              hyperparameters,
              actor_model=args.actor_model,
              critic_model=args.critic_model,
              decision_model=args.decision_model,
              class_dim=args.class_dim)
    else:
        test(env,
             actor_model=args.actor_model,
             class_dim=args.class_dim)


if __name__ == '__main__':
    args = get_args()
    main(args)