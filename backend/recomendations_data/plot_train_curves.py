import matplotlib.pyplot as plt
import csv

episodes = []
rewards = []
losses = []

with open("logs/training_logs.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        episodes.append(int(row["Episode"]))
        rewards.append(float(row["Reward"]))
        losses.append(float(row["Loss"]))

# Plotting Reward
plt.figure(figsize=(10, 5))
plt.plot(episodes, rewards, label="Total Reward per Episode", color="green")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training Reward Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("reward_curve.png")
plt.show()

# Plotting Loss
plt.figure(figsize=(10, 5))
plt.plot(episodes, losses, label="Average Loss per Episode", color="red")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("DQN Training Loss Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()
