import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical    # 提供了一组概率分布的实现，这里用于表示离散类别型分布

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 环境参数
TARGET_RANGE = (4.0, 6.0)
INITIAL_HEIGHT = 10.0
X_BOUNDS = (0.0, 10.0)


class DropBlockEnv:
    def __init__(self):
        self.x = None
        self.y = None
        self.reset()

    def reset(self):
        self.y = INITIAL_HEIGHT    # 10.0
        self.x = np.random.uniform(X_BOUNDS[0], X_BOUNDS[1])    # (0.0, 10.0)
        return self._get_state()

    def _get_state(self):
        return np.array([self.x / X_BOUNDS[1], self.y / INITIAL_HEIGHT], dtype=np.float32)

    def step(self, action):
        new_x = self.x + (1.0 if action else -1.0)
        new_x = np.clip(new_x, X_BOUNDS[0], X_BOUNDS[1])
        self.y -= 1.0
        self.x = new_x

        done = self.y <= 0
        reward = self._calculate_reward(done)
        return self._get_state(), reward, done, {}

    def _calculate_reward(self, done):
        if done:
            return 10.0 if TARGET_RANGE[0] <= self.x <= TARGET_RANGE[1] else -10.0

        distance = abs(self.x - np.mean(TARGET_RANGE))
        in_target_air = TARGET_RANGE[0] <= self.x <= TARGET_RANGE[1]
        reward = 0.2 if in_target_air else -0.1
        reward -= 0.05 * distance + 0.1
        return reward


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2,
                 epochs=4, batch_size=64, critic_lr=3e-4):
        # print(state_dim): 2    # 状态空间的维度
        # print(action_dim): 2    # 动作空间的维度
        # print(lr): 0.0003    # Actor网络学习率
        # print(gamma): 0.99    # 折扣累积回报
        # print(epsilon): 0.2    # PPO-Clip，限制概率比值
        # print(epochs): 4    # 利用旧的采样数据，更新参数四次
        # print(batch_size): 64    # 每次更新参数，采样64个状态动作对
        # print(critic_lr): 0.0003    # Critic网络学习率

        self.actor = Actor(state_dim, action_dim).to(device)    # 2, 2
        self.critic = Critic(state_dim).to(device)    # 2
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)    # 0.0003
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)    # 0.0003
        self.gamma = gamma    # 0.99
        self.epsilon = epsilon    # 0.2
        self.epochs = epochs    # 4
        self.batch_size = batch_size    # 64

    def update(self, states, actions, old_probs, rewards, dones):
        # print(len(states)): 100
        # print(len(actions)): 100
        # print(len(old_probs)): 100
        # print(len(rewards)): 100
        # print(len(dones)): 100

        # 计算蒙特卡洛回报, 即累积折扣汇报
        # print(3.2 * True, 3.2 * False): 3.2 0.0
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + self.gamma * G * (not done)    # False为0，True为1
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        # print(returns):
        # tensor([  7.6030,   8.0589,   8.4690,   8.8327,   8.8465,   9.2140,   9.2317, 9.6030,   9.6246,  10.0000,
        #         -11.8180, -11.6687, -11.4674, -11.2135, -11.0076, -10.8501, -10.6405, -10.4793, -10.2660, -10.0000,
        #         ......
        #         -11.8949, -11.5605, -11.2732, -11.0336, -10.8420, -10.5980, -10.3010, -10.0515, -9.8500, -10.0000])

        # 转换数据为tensor
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
        old_probs_tensor = torch.tensor(old_probs, dtype=torch.float32).to(device)
        # print(states_tensor.shape): torch.Size([100, 2])
        # print(actions_tensor.shape): torch.Size([100])
        # print(old_probs_tensor.shape): torch.Size([100])

        # 计算状态值, 计算优势值, 并标准化优势
        with torch.no_grad():
            values = self.critic(states_tensor)
            # print(values.shape): torch.Size([100])
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # print(advantages.shape): torch.Size([100])

        # 训练过程
        for _ in range(self.epochs):    # 4
            indices = np.arange(len(states))    # 100
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                batch_states = states_tensor[idx]
                batch_actions = actions_tensor[idx]
                batch_old_probs = old_probs_tensor[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                # print(batch_states.shape): torch.Size([64, 2])
                # print(batch_actions.shape): torch.Size([64])
                # print(batch_old_probs.shape): torch.Size([64])
                # print(batch_advantages.shape): torch.Size([64])
                # print(batch_returns.shape): torch.Size([64])

                # 更新Actor
                new_probs = self.actor(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze()
                # print(new_probs.shape): torch.Size([64])
                ratio = new_probs / batch_old_probs
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                # print(ratio.shape): torch.Size([64])
                # print(surr1.shape): torch.Size([64])
                # print(surr2.shape): torch.Size([64])

                actor_loss = -torch.min(surr1, surr2).mean()
                # print(actor_loss): tensor(0.0662, device='cuda:0', grad_fn=<NegBackward0>)

                # 更新Critic
                predicted_values = self.critic(batch_states)
                critic_loss = nn.MSELoss()(predicted_values, batch_returns)
                # print(predicted_values.shape): torch.Size([64])
                # print(critic_loss): tensor(115.8767, device='cuda:0', grad_fn=<MseLossBackward0>)

                """ 震惊了，突然意识到Actor优化器、Critic优化器其实可以分别梯度回传，更新各自的参数!!! """
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optim.step()
                self.critic_optim.step()


def evaluate(policy, num_episodes=1000):
    env = DropBlockEnv()
    safe_landings = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action_probs = policy.actor(state_tensor)
                action = torch.argmax(action_probs).item()

            next_state, _, done, _ = env.step(action)
            state = next_state

        final_x = state[0] * X_BOUNDS[1]
        if TARGET_RANGE[0] <= final_x <= TARGET_RANGE[1]:
            safe_landings += 1

    return safe_landings / num_episodes


def train():
    env = DropBlockEnv()
    state_dim = 2
    action_dim = 2
    ppo = PPO(state_dim, action_dim)    # 2, 2

    num_episodes_per_update = 10    # 每次采样10条轨迹数据
    total_updates = 2501
    eval_interval = 10

    success_rate = evaluate(ppo, num_episodes=10000)
    print(f"Initial Evaluation Safe Rate: {success_rate:.4f}")

    for update in range(total_updates):    # 执行2501次，每次采样10条轨迹数据，并使用这批数据更新4次参数
        states, actions, old_probs, rewards, dones = [], [], [], [], []

        # 收集数据
        for _ in range(num_episodes_per_update):    # 10
            state = env.reset()
            # print(state): [0.9507143 1.       ]
            done = False

            while not done:
                state_tensor = torch.FloatTensor(state).to(device)
                # print(state_tensor): tensor([0.9507, 1.0000], device='cuda:0')
                with torch.no_grad():
                    action_probs = ppo.actor(state_tensor)
                    # print(action_probs): tensor([0.5161, 0.4839], device='cuda:0')
                    dist = Categorical(action_probs)
                    # print(dist): Categorical(probs: torch.Size([2]))
                    action = dist.sample()
                    # print(action): tensor(0, device='cuda:0')
                    old_prob = action_probs[action.item()].item()
                    # print(old_prob): 0.5160935521125793

                next_state, reward, done, _ = env.step(action.item())
                # print(next_state): [0.8507143 0.9      ]
                # print(reward): -0.375357153204958
                # print(done): False
                # print(_): {}

                states.append(state)
                actions.append(action.item())
                old_probs.append(old_prob)
                rewards.append(reward)
                dones.append(done)

                state = next_state

        # 更新策略
        ppo.update(states, actions, old_probs, rewards, dones)

        # 评估性能
        if (update + 1) % eval_interval == 0:
            success_rate = evaluate(ppo, 1000)
            print(f"Update {update + 1}, Success Rate: {success_rate:.3f}")

    success_rate = evaluate(ppo, num_episodes=10000)
    print(f"Final Evaluation Safe Rate: {success_rate:.4f}")
    torch.save(ppo.actor.state_dict(), 'policy_MC.pth')


if __name__ == "__main__":
    train()

    """
    Initial Evaluation Safe Rate: 0.1484
    Update 10, Success Rate: 0.882
    Update 20, Success Rate: 0.514
    Update 30, Success Rate: 0.613
    Update 40, Success Rate: 0.614
    Update 50, Success Rate: 0.988
    Update 60, Success Rate: 0.803
    Update 70, Success Rate: 0.977
    Update 80, Success Rate: 0.979
    Update 90, Success Rate: 0.962
    Update 100, Success Rate: 0.895
    
    Update 2310, Success Rate: 0.974
    Update 2320, Success Rate: 0.976
    Update 2330, Success Rate: 0.963
    Update 2340, Success Rate: 0.929
    Update 2350, Success Rate: 1.000
    Update 2360, Success Rate: 0.952
    Update 2370, Success Rate: 0.965
    Update 2380, Success Rate: 0.966
    Update 2390, Success Rate: 0.961
    Update 2400, Success Rate: 0.991
    Update 2410, Success Rate: 0.997
    Update 2420, Success Rate: 0.992
    Update 2430, Success Rate: 0.946
    Update 2440, Success Rate: 0.970
    Update 2450, Success Rate: 0.968
    Update 2460, Success Rate: 0.998
    Update 2470, Success Rate: 0.937
    Update 2480, Success Rate: 0.987
    Update 2490, Success Rate: 0.963
    Update 2500, Success Rate: 0.971
    Final Evaluation Safe Rate: 0.9828
    """