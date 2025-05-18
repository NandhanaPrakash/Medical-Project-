import torch
import numpy as np
import pandas as pd
from dqn_agent import DQNAgent
from ehr_env import EHR_Env
import os
import csv

# ✅ Load dataset
data = pd.read_csv("final_ehr_dataset_cleaned.csv")
num_total = len(data)
num_test = int(num_total * 0.2)  # Last 20% as test patients
unseen_data = data.iloc[-num_test:].reset_index(drop=True)

# ✅ Save unseen data for reference
unseen_data.to_csv("logs/unseen_patients.csv", index=False)

# ✅ Init environment on unseen data
env = EHR_Env(data_path="logs/unseen_patients.csv")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# ✅ Load trained agent
agent = DQNAgent(state_dim, action_dim)
agent.load("checkpoints/dqn_final.pth")
agent.epsilon = 0.0  # Fully greedy for evaluation

# ✅ Log results
results = []

for i in range(len(unseen_data)):
    state = env.reset(patient_index=i)
    total_reward = 0
    actions_taken = []
    bmis, glucoses = [], []

    for _ in range(env.max_steps):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        actions_taken.append(action)
        bmis.append(info["new_bmi"])
        glucoses.append(info["new_glucose"])

        state = next_state
        if done:
            break

    results.append({
        "Patient": i,
        "Initial_BMI": bmis[0],
        "Final_BMI": bmis[-1],
        "Initial_Glucose": glucoses[0],
        "Final_Glucose": glucoses[-1],
        "Total_Reward": total_reward,
        "Actions_Taken": actions_taken
    })

# ✅ Save evaluation logs
os.makedirs("logs", exist_ok=True)
with open("logs/unseen_patient_evaluation.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("🧪 Evaluation complete on unseen patients.")
print("📄 Saved to logs/unseen_patient_evaluation.csv")
