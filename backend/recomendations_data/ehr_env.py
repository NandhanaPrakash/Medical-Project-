import gym
import numpy as np
import pandas as pd
import random
from gym import spaces

class EHR_Env(gym.Env):
    def __init__(self, data_path):
        super(EHR_Env, self).__init__()
        self.data = pd.read_csv(data_path)
        self.original_data = self.data.copy()
        self.patient_index = 0
        self.current_step = 0
        self.max_steps = 100

        self.action_meanings = {
            0: "Low Carb Diet",
            1: "High Protein Diet",
            2: "Mediterranean Diet",
            3: "Vegetarian Diet",
            4: "Intermittent Fasting",
            5: "Walking 30 min/day",
            6: "Jogging 20 min/day",
            7: "Strength Training",
            8: "Yoga and Flexibility",
            9: "High Intensity Interval Training"
        }

        self.feature_cols = [col for col in self.data.columns 
                             if col != 'SEQN' and self.data[col].dtype in [np.float64, np.int64]]
        self.num_features = len(self.feature_cols)
        self._preprocess()

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_meanings))
        self.reset()

    def _preprocess(self):
        activity_map = {'Low': 200, 'Moderate': 400, 'High': 600}
        if 'Activity_Level' in self.data.columns:
            self.data['Activity_Level'] = self.data['Activity_Level'].map(activity_map).fillna(300)

        self.data[self.feature_cols] = self.data[self.feature_cols].apply(pd.to_numeric, errors='coerce')
        self.data.fillna(0, inplace=True)

        self.feature_min = self.data[self.feature_cols].min()
        self.feature_max = self.data[self.feature_cols].max()

        self.data[self.feature_cols] = (
            self.data[self.feature_cols] - self.feature_min
        ) / (self.feature_max - self.feature_min + 1e-8)

        self.original_data = self.data.copy()

    def _get_observation(self):
        obs = self.data.iloc[self.patient_index][self.feature_cols].values.astype(np.float32)
        return np.clip(obs, 0, 1)

    def _compute_bmi(self, patient_data):
        kcal1 = patient_data.get('DR1TKCAL', 0)
        kcal2 = patient_data.get('DR2TKCAL', 0)
        activity_burn = patient_data.get('Activity_Level', 300)

        total_calories = np.clip(kcal1 + kcal2, 1000, 6000)
        activity_burn = np.clip(activity_burn, 100, 1000)

        bmi = 22 + 0.0007 * total_calories - 0.0008 * activity_burn
        return np.clip(bmi, 15, 40)

    def _compute_glucose(self, patient_data):
        sugar1 = patient_data.get('DR1TSUGR', 0)
        sugar2 = patient_data.get('DR2TSUGR', 0)
        activity_burn = patient_data.get('Activity_Level', 300)

        total_sugar = np.clip(sugar1 + sugar2, 0, 300)
        activity_burn = np.clip(activity_burn, 100, 1000)

        glucose = 90 + 0.15 * total_sugar - 0.02 * activity_burn
        return np.clip(glucose, 65, 160)

    def _calculate_reward(self, prev_bmi, new_bmi, prev_glucose, new_glucose):
        delta_bmi = prev_bmi - new_bmi
        delta_glucose = prev_glucose - new_glucose

        reward = 0.4 * delta_bmi + 0.6 * delta_glucose

        if abs(new_bmi - 22) < 1.5:
            reward += 1.5
        if 80 <= new_glucose <= 110:
            reward += 2.0

        return np.clip(reward, -10, 15)

    def step(self, action):
        self.current_step += 1
        patient_data = self.original_data.iloc[self.patient_index].copy()

        # Personalized response scaling
        diet_sensitivity = np.random.uniform(0.95, 0.99)
        exercise_sensitivity = np.random.uniform(1.05, 1.15)

        prev_bmi = self._compute_bmi(patient_data)
        prev_glucose = self._compute_glucose(patient_data)

        # Apply diet intervention
        if action in [0, 1, 2, 3, 4]:
            for kcal_col in ['DR1TKCAL', 'DR2TKCAL']:
                if kcal_col in self.original_data.columns:
                    self.original_data.at[self.patient_index, kcal_col] *= diet_sensitivity
            for sugar_col in ['DR1TSUGR', 'DR2TSUGR']:
                if sugar_col in self.original_data.columns:
                    self.original_data.at[self.patient_index, sugar_col] *= (diet_sensitivity - 0.02)

        # Apply exercise intervention
        if action in [5, 6, 7, 8, 9]:
            if 'Activity_Level' in self.original_data.columns:
                self.original_data.at[self.patient_index, 'Activity_Level'] *= exercise_sensitivity

        # Recompute outcomes
        new_data = self.original_data.iloc[self.patient_index]
        new_bmi = self._compute_bmi(new_data)
        new_glucose = self._compute_glucose(new_data)

        reward = self._calculate_reward(prev_bmi, new_bmi, prev_glucose, new_glucose)
        done = self.current_step >= self.max_steps
        obs = self._get_observation()

        info = {
            "action_index": action,
            "action_name": self.action_meanings[action],  # ✅ Add this line
            'prev_bmi': round(prev_bmi, 2),
            'new_bmi': round(new_bmi, 2),
            'prev_glucose': round(prev_glucose, 2),
            'new_glucose': round(new_glucose, 2),
            'reward': round(reward, 2)
        }

        return obs, reward, done, info

    def reset(self, patient_index=None):
        self.current_step = 0
        self.done = False

        if patient_index is not None:
            self.patient_index = patient_index
            self.current_patient = self.data.iloc[patient_index].copy()
        else:
            sample = self.data.sample(1)
            self.patient_index = sample.index[0] 
            self.current_patient = self.data.sample(1).iloc[0].copy()

        self.original_bmi = self._compute_bmi(self.current_patient)
        self.original_glucose = self._compute_glucose(self.current_patient)

        return self._get_observation()



    def render(self, mode='human'):
        print(f"Patient Index: {self.patient_index} | Step: {self.current_step}")
