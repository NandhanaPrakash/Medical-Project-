import gym
from gym import spaces
import numpy as np
import pandas as pd

class EHRDietExerciseEnv(gym.Env):
    def __init__(self, ehr_csv_path):
        super(EHRDietExerciseEnv, self).__init__()

        # Load EHR dataset
        self.data = pd.read_csv(ehr_csv_path)
        self.num_patients = self.data.shape[0]
        self.current_index = 0  # Track patient index

        # Define observation space (EHR data)
        self.observation_space = spaces.Box(
            low=0, high=5000, shape=(self.data.shape[1]-1,), dtype=np.float32
        )

        # Define action space (5 diet & 5 exercise choices)
        self.action_space = spaces.Discrete(10)

    def reset(self):
        """ Reset environment to start new episode """
        self.current_index = np.random.randint(0, self.num_patients)
        patient_data = self.data.iloc[self.current_index, 1:].values
        return patient_data.astype(np.float32)

    def step(self, action):
        """ Take an action and return the new state, reward, and done flag """
        # Reward logic: Encourage balanced nutrition and physical activity
        reward = self.calculate_reward(action)

        # Move to next patient or terminate episode
        done = np.random.rand() > 0.9
        next_state = self.reset() if not done else np.zeros_like(self.data.iloc[0, 1:].values)
        
        return next_state, reward, done, {}

    def calculate_reward(self, action):
        """ Reward function based on recommended diet & exercise """
        if action in [0, 1, 2, 3, 4]:  # Diet choices
            return np.random.uniform(0, 1)  # Random positive reward
        elif action in [5, 6, 7, 8, 9]:  # Exercise choices
            return np.random.uniform(0, 2)  # Higher reward for exercise
        return -1  # Default negative reward

