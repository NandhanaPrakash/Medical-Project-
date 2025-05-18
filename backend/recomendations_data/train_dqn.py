import numpy as np
import torch
from dqn_agent import DQNAgent
from ehr_env import EHR_Env
import os
import matplotlib.pyplot as plt
import csv

# ✅ Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ✅ Initialize environment and agent
env = EHR_Env(data_path='final_ehr_dataset_cleaned.csv')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)

# ✅ Training parameters
num_episodes = 300
batch_size = 64
reward_clip = (-1000, 1000)  # prevent extreme rewards

# ✅ Logging
rewards_list, losses, bmis, glucoses = [], [], [], []

for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0
    episode_bmis, episode_glucoses = [], []

    for step in range(env.max_steps):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        # ✅ Clip extreme rewards
        reward = np.clip(reward, *reward_clip)

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        episode_bmis.append(info['new_bmi'])
        episode_glucoses.append(info['new_glucose'])

        if done:
            break

    # ✅ Train the agent
    avg_loss = agent.replay(batch_size)
    losses.append(avg_loss)

    # ✅ Log episode metrics
    rewards_list.append(total_reward)
    bmis.append(np.mean(episode_bmis))
    glucoses.append(np.mean(episode_glucoses))

    print(f"[Ep {episode}/{num_episodes}] 🔁 Steps: {env.max_steps} | 🎯 Reward: {total_reward:.2f} "
          f"| 📉 Avg Loss: {avg_loss:.4f} | 🧍 BMI: {np.mean(episode_bmis):.2f} "
          f"| 🩸 Glucose: {np.mean(episode_glucoses):.2f} | 🔽 Epsilon: {agent.epsilon:.3f}")

    # ✅ Save checkpoint
    if episode % 50 == 0 or episode == num_episodes:
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/dqn_episode_{episode}.pth"
        agent.save(path)
        print(f"💾 Saved model checkpoint at: {path}")

# ✅ Final model save
agent.save("checkpoints/dqn_final.pth")
print("✅ Training completed. Final model saved to checkpoints/dqn_final.pth")

# ✅ Save metrics to CSV
os.makedirs("logs", exist_ok=True)
with open("logs/training_logs.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Reward", "Loss", "BMI", "Glucose"])
    for i in range(num_episodes):
        writer.writerow([i+1, rewards_list[i], losses[i], bmis[i], glucoses[i]])

# ✅ Plotting
os.makedirs("plots", exist_ok=True)

def plot_and_save(values, title, ylabel, filename):
    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}")
    plt.close()

plot_and_save(rewards_list, "Episode Rewards", "Reward", "rewards_plot.png")
plot_and_save(losses, "Loss per Episode", "Loss", "loss_plot.png")
plot_and_save(bmis, "Average BMI per Episode", "BMI", "bmi_plot.png")
plot_and_save(glucoses, "Average Glucose per Episode", "Glucose", "glucose_plot.png")

print("📊 Training plots saved in 'plots/' folder.")
print("📁 Episode metrics saved in 'logs/training_logs.csv'")
