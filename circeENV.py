from typing import Optional
import numpy as np
import gymnasium as gym


class TheusEnv(gym.Env):
    """
    			Class used to simulate the environment of electrical faults in hybrid networks.

    			Parameters:
    				batch_charact: Database for training (Dataloader).
    				n_charact: Number of characters used for training.

    			Returns:
    				None
    		"""

    def __init__(self, batch_charact,
                 n_charact: int=10):
        super().__init__()

        self.membership = None
        self.total = 0

        if batch_charact:
            self.dataloader = iter(batch_charact)
            self.batch = batch_charact.batch_size

        self.current_data = None
        self.current_batch = None
        self.index_batch = 0

        self.n_charact = n_charact

        # ACTIONS
        # Init to -1
        self.p1 = np.full(self.n_charact, -1.0)
        self.pc = np.full(self.n_charact, -1.0)
        self.p2 = np.full(self.n_charact, -1.0)

        self.n_actions = n_charact
        self.action_space = gym.spaces.Dict(
            {
                "p1": gym.spaces.Box(low=-1, high=1, shape=(self.n_actions,)),
                "pc": gym.spaces.Box(low=-1, high=1, shape=(self.n_actions,)),
                "p2": gym.spaces.Box(low=-1, high=1, shape=(self.n_actions,)),
            }
        )

        # ONSERVATIONS
        self._agent_charact = None

        self.n_states = n_charact
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(n_charact,), dtype=np.float32)

    def get_labels(self):
        """
            Gets labels from the database loaded

            Parameters:
                None

            Returns:
                Tensor of labels with size (batch,)
        """
        return self.current_batch[1][self.index_batch]

    def _get_obs(self):
        """
            Gets current observation from the database loaded

            Parameters:
                None

            Returns:
                Ndarray of observations with size (batch,)
        """
        self._agent_charact = self.current_batch[0][self.index_batch].detach().numpy()
        return self._agent_charact

    def _new_obs(self):
        """
            Gets a new observation from the database loaded, and the index in batch is incremented

            Parameters:
                None

            Returns:
                Ndarray of observations with size (batch,)
        """
        self.index_batch += 1
        self._agent_charact = self.current_batch[0][self.index_batch].detach().numpy()
        return self._agent_charact

    def _get_action(self):
        """
            Gets the actions as three vector, p1, pc and p2

            Parameters:
                None

            Returns:
                Dictionary of vectors p1, pc and p2
        """
        return {"p1": self.p1, "pc": self.pc, "p2": self.p2}

    def _get_info(self):
        """
            Gets the performance of the decision process. Just in this version

            Parameters:
                None

            Returns:
                Accuracy of the decision process
        """
        return {
            "accuracy": self._agent_acc,
        }

    def get_membership(self):
        """
            Gets the membership of the current observation

            Parameters:
                None

            Returns:
                Membership of the current observation
        """
        return self.membership


    def encoder(self):
        """
            Gets the fuzzification of the current observation, and saves it in self.membership
            as a ndarray

            Parameters:
                None

            Returns:
                None
        """
        """ Codifica cada uno de las características: Fuzzification
        Devuelve:
            np.Array (batch x num characteristics)
        """
        trifuzz = crear_fuzzificacion_triangulos(self.p1, self.pc, self.p2, size=self.n_charact)

        data = self.current_data.squeeze()
        self.membership = np.zeros(self.n_charact)
        for c in range(self.n_charact):
            self.membership[c] = trifuzz[c](data[c])


    def init_rollout(self):
        """
            Init the rollout to get the simulation results for each observation.
            Init the dataloader and reset the value of self.index_batch

            Parameters:
                None

            Returns:
                None
        """
        try:
            self.current_batch = next(self.dataloader)
        except StopIteration:
            iterator = iter(self.dataloader)
            self.total = 0
        self.index_batch = 0


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """"
            Fill with ramdom values the vectors p1, pc and p2. Fill the first observation of the batch.

            Parameters:
                None

            Returns:
                Returns the first obsrvation and the info (accuracy)
        """
        super().reset(seed=seed)

        # TODO: I am sure if this is needed since it is already done in init_rollout
        self.index_batch = 0

        # Damos valores aleatorios a los vectores de salida asegurando que p1<pc<p2
        for i in range(self.n_charact):
            # Generar tres valores aleatorios y ordenarlos
            valores = np.random.rand(3)
            self.p1[i], self.pc[i], self.p2[i] = np.sort(valores)

            # Si alguno es igual, forzar desigualdad mínima sumando un delta pequeño
            if self.p1[i] == self.pc[i]:
                self.pc[i] = self.p1[i] + 1e-6
            if self.pc[i] == self.p2[i]:
                self.p2[i] = self.pc[i] + 1e-6
            # Asegurarse que no exceda 1
            self.p2[i] = min(self.p2[i], 1.0)

        # Chequeamos que los 3 vectores de acción están bien ordenados
        assert self.p1.shape == self.pc.shape == self.p2.shape, "Los arrays deben tener la misma forma"
        assert np.all(self.p1 < self.pc), "Algunos elementos no cumplen p1[i] < pc[i]"
        assert np.all(self.pc < self.p2), "Algunos elementos no cumplen pc[i] < p2[i]"

        # Dar valor a la observación
        self._agent_acc = np.random.rand()
        #self._agent_charact = self.current_batch[0][self.index_batch]#np.random.rand(self.n_charact)

        self.total += self.current_batch[0].shape[0]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment and gets a new observation of the batch.

        Args:
            action: The action to take (p1, pc, p2)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        try:
            self.current_data = self.current_batch[0][self.index_batch]

            self.p1 = action[0, 0:self.n_actions]
            self.pc = action[0, self.n_actions: 2 * self.n_actions]
            self.p2 = action[0, 2 * self.n_actions: 3 * self.n_actions]

            self.encoder()

            #"No usamos el criterio para truncar el episodio"
            truncated = False
            #"Fijamos la terminación del periodo con terminated"
            terminated = False
            if self.index_batch == self.batch - 1:
                terminated = True
            #"Obtenemos la recompensa basada en el accuracy obtenido"
            reward = self._agent_acc
            #"Recogemos la exactitud actual conseguida y obtenemos una nueva obs"
            observation = self._new_obs()
            info = self._get_info()
        except StopIteration:
            observation = None
            reward = 0
            terminated = True
            truncated = False
            info = {}

        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass
        #print(self._agent_acc)


"""----------------------------------------------------------------"""
def crear_fuzzificacion_triangulos(inicios, picos, fines, size):
    """
    Crea 10 funciones de pertenencia triangulares para lógica difusa.
    Cada función define un triángulo con sus puntos de inicio (0), pico (1), y fin (0).

    Parámetros:
        inicios: array numpy de SIZE elementos, inicio del triángulo (donde y=0)
        picos: array numpy de SIZE elementos, pico del triángulo (donde y=1)
        fines: array numpy de SIZE elementos, fin del triángulo (donde y=0)

    Retorna:
        Lista de 10 funciones lambda, cada una que toma x y devuelve el grado de pertenencia.
    """
    funciones = []
    for i in range(size):
        a, b, c = inicios[i], picos[i], fines[i]

        # Protección ante división por cero para triángulos degenerados
        def triangular(x, a=a, b=b, c=c):
            if x <= a or x >= c:
                return 0.0
            elif a < x < b:
                return (x - a) / (b - a) if b != a else 1.0
            elif b < x < c:
                return (c - x) / (c - b) if c != b else 1.0
            else:  # x == b
                return 1.0

        funciones.append(triangular)
    return funciones
