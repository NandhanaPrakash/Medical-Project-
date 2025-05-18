import numpy as np
import torch
from dqn_agent import DQNAgent
from ehr_env import EHR_Env

# ✅ Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ✅ Load environment and agent
env = EHR_Env(data_path='final_ehr_dataset_cleaned.csv')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)
agent.load("checkpoints/dqn_final.pth")  # Load trained model
agent.epsilon = 0.0  # Pure exploitation (no exploration)

# ✅ Evaluation parameters
num_eval_episodes = 20
all_rewards, all_bmis, all_glucoses = [], [], []

print("\n🚀 Starting Evaluation...\n")

for ep in range(1, num_eval_episodes + 1):
    state = env.reset()
    total_reward = 0
    episode_bmis, episode_glucoses, episode_actions = [], [], []

    for step in range(env.max_steps):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        state = next_state
        total_reward += reward

        episode_bmis.append(info['new_bmi'])
        episode_glucoses.append(info['new_glucose'])
        episode_actions.append(info['action_name'])  # readable diet/exercise combo

        if done:
            break

    avg_bmi = np.mean(episode_bmis)
    avg_glucose = np.mean(episode_glucoses)
    all_rewards.append(total_reward)
    all_bmis.append(avg_bmi)
    all_glucoses.append(avg_glucose)

    print(f"[Eval Ep {ep}/{num_eval_episodes}] 🎯 Total Reward: {total_reward:.2f} "
          f"| 🧍 Avg BMI: {avg_bmi:.2f} | 🩸 Avg Glucose: {avg_glucose:.2f}")
    print(f"🔄 Actions Taken: {episode_actions}\n")

# ✅ Summary
print("\n📊 Evaluation Summary:")
print(f"🏅 Avg Reward: {np.mean(all_rewards):.2f}")
print(f"🧍 Avg BMI: {np.mean(all_bmis):.2f}")
print(f"🩸 Avg Glucose: {np.mean(all_glucoses):.2f}")
