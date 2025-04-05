import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
            nn.Linear(state_dim, hidden_size),    # (2, 128)
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),    # (128, 128)
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),    # (128, 2)
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),    # (2, 128)
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),    # (128, 128)
            nn.ReLU(),
            nn.Linear(hidden_size, 1)    # (128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2,
                 epochs=4, batch_size=64, critic_lr=3e-4):
        # print(state_dim): 2    # 状态空间的维度
        # print(action_dim): 2    # 动作空间的维度
        # print(lr): 0.0003    # Actor网络学习率
        # print(gamma): 0.99    # 累积折扣回报
        # print(epsilon): 0.2    # PPO-Clip，限制概率比值
        # print(epochs): 4    # 利用旧的采样数据，更新参数四次
        # print(batch_size): 64    # 每次更新参数，采样64个状态动作对
        # print(critic_lr): 0.0003     # Critic网络学习率

        self.actor = Actor(state_dim, action_dim).to(device)    # 2 2
        self.critic = Critic(state_dim).to(device)    # 2
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)    # 0.0003
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)    # 0.0003
        self.gamma = gamma    # 0.99
        self.epsilon = epsilon    # 0.2
        self.epochs = epochs    # 4
        self.batch_size = batch_size    # 64

    def update(self, states, actions, old_probs, rewards, dones, next_states):
        # print(len(states)): 100
        # print(len(actions)): 100
        # print(len(old_probs)): 100
        # print(len(rewards)): 100
        # print(len(dones)): 100
        # print(len(next_states)): 100

        # 转换数据为tensor
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        # print(states_tensor.shape): torch.Size([100, 2])
        # print(next_states_tensor.shape): torch.Size([100, 2])

        # 使用时序差分（TD）计算回报
        with torch.no_grad():
            values = self.critic(states_tensor)
            next_values = self.critic(next_states_tensor)
            # print(values.shape): torch.Size([100])
            # print(next_values.shape): torch.Size([100])

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool).to(device)
        # print(rewards_tensor.shape): torch.Size([100])
        # print(dones_tensor.shape): torch.Size([100])
        # print(rewards_tensor):
        # tensor([ -0.3754,  -0.3254,  -0.2754,   0.0746,  -0.2754,   0.0746,  -0.2754, 0.0746,  -0.2754,  10.0000,
        #          -0.2660,  -0.3160,  -0.3660,  -0.3160,  -0.2660,  -0.3160,  -0.2660,  -0.3160,  -0.3660, -10.0000,
        #          -0.2993,  -0.3493,  -0.2993,  -0.3493,  -0.2993,   0.0507,  -0.2993,   0.0507,  -0.2993, -10.0000,
        #          -0.4220,  -0.3720,  -0.4220,  -0.4500,  -0.4500,  -0.4000,  -0.3500,  -0.3000,  -0.3500, -10.0000,
        #          -0.4220,  -0.4500,  -0.4000,  -0.3500,  -0.4000,  -0.4500,  -0.4000,  -0.3500,  -0.4000, -10.0000,
        #          -0.4500,  -0.4500,  -0.4500,  -0.4000,  -0.4500,  -0.4500,  -0.4500,  -0.4500,  -0.4500, -10.0000,
        #          -0.4331,  -0.3831,  -0.3331,  -0.3831,  -0.3331,  -0.2831,   0.0669,   0.0831,  -0.2669, -10.0000,
        #          -0.3006,  -0.2506,  -0.3006,  -0.3506,  -0.4006,  -0.3506,  -0.3006,  -0.3506,  -0.3006, -10.0000,
        #          -0.2540,   0.0960,  -0.2540,  -0.3040,  -0.3540,  -0.4040,  -0.3540,  -0.3040,  -0.2540,  10.0000,
        #          -0.4500,  -0.4000,  -0.3500,  -0.3000,  -0.3500,  -0.4000,  -0.3500,  -0.3000,  0.0500, -10.0000], )

        # False为0，True为1
        """ 在（蒙特卡洛）MC方法中，将当前步以及之后的累积折扣奖励，作为Q(s, a), 即动作值函数;
            而在（时序差分）TD方法中, 将当前步的奖励, 加上下一步的状态值乘以gamma, 作为Q(s, a), 即动作值函数,
            其实这种定义也挺合理, 状态值表示的是当前状态下所有可能动作的平均Q(s, a)， 下一步的状态值就表示下一步状态下所有可能动作的平均Q(s, a)，
            再加上当前动作的奖励，正好用于定义当前动作的动作值函数Q(s, a), 非常合理.
            """
        returns = rewards_tensor + self.gamma * next_values * (~dones_tensor)
        # print(returns.shape): torch.Size([100])

        # 计算优势值并标准化
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # print(advantages.shape): torch.Size([100])

        # 转换其他数据
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
        old_probs_tensor = torch.tensor(old_probs, dtype=torch.float32).to(device)
        # print(actions_tensor.shape): torch.Size([100])
        # print(old_probs_tensor.shape): torch.Size([100])

        # 训练过程
        for _ in range(self.epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                batch_states = states_tensor[idx]
                batch_actions = actions_tensor[idx]
                batch_old_probs = old_probs_tensor[idx]
                batch_advantages = advantages[idx]
                # print(batch_states.shape): torch.Size([64, 2])
                # print(batch_actions.shape): torch.Size([64])
                # print(batch_old_probs.shape): torch.Size([64])
                # print(batch_advantages.shape): torch.Size([64])

                # 更新Actor
                new_probs = self.actor(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze()
                ratio = new_probs / batch_old_probs
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                # print(new_probs.shape): torch.Size([64])
                # print(ratio.shape): torch.Size([64])
                # print(surr1.shape): torch.Size([64])
                # print(surr2.shape): torch.Size([64])

                actor_loss = -torch.min(surr1, surr2).mean()
                # print(actor_loss): tensor(-0.0952, device='cuda:0', grad_fn=<NegBackward0>)

                """
                Advantage(at, st) = Q(at, st) - V(st)
                蒙特卡洛法、时序差分法都是在Q(at, st)公式上进行了变形, 在监督Critic网络的估计值时，依旧使用Q(at, st)作为Ground Truth
                """

                # 更新Critic
                predicted_values = self.critic(batch_states)
                critic_loss = nn.MSELoss()(predicted_values, returns[idx])
                # print(predicted_values.shape): torch.Size([64])
                # print(critic_loss): tensor(9.5806, device='cuda:0', grad_fn=<MseLossBackward0>)

                # 梯度更新
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
    ppo = PPO(state_dim, action_dim)    # 2 2

    num_episodes_per_update = 10    # 每次更新时，采样10条轨迹数据
    total_updates = 2501    # 总共进行2501次更新（每次更新时，会基于旧的轨迹数据，对参数优化更新4次）
    eval_interval = 10

    success_rate = evaluate(ppo, num_episodes=10000)
    print(f"Initial Evaluation Safe Rate: {success_rate:.4f}")

    for update in range(total_updates):    # 执行2501次，每次采样10条轨迹数据，并使用这批数据更新4次参数
        states, actions, old_probs, rewards, dones, next_states = [], [], [], [], [], []

        # 收集数据（新增next_states收集）
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
                next_states.append(next_state)  # 新增next_states收集
                actions.append(action.item())
                old_probs.append(old_prob)
                rewards.append(reward)
                dones.append(done)

                state = next_state

        # 更新策略（传入next_states）
        ppo.update(states, actions, old_probs, rewards, dones, next_states)

        # 评估性能
        if (update + 1) % eval_interval == 0:
            success_rate = evaluate(ppo, 1000)
            print(f"Update {update + 1}, Success Rate: {success_rate:.3f}")

    success_rate = evaluate(ppo, num_episodes=10000)
    print(f"Final Evaluation Safe Rate: {success_rate:.4f}")
    torch.save(ppo.actor.state_dict(), 'policy_TD.pth')


if __name__ == "__main__":
    train()

    """
    Initial Evaluation Safe Rate: 0.1484
    Update 10, Success Rate: 0.000
    Update 20, Success Rate: 0.874
    Update 30, Success Rate: 0.791
    Update 40, Success Rate: 0.939
    Update 50, Success Rate: 0.958
    Update 60, Success Rate: 0.896
    Update 70, Success Rate: 0.917
    Update 80, Success Rate: 0.999
    Update 90, Success Rate: 0.917
    Update 100, Success Rate: 0.982
    ... ...
    Update 2400, Success Rate: 0.996
    Update 2410, Success Rate: 0.997
    Update 2420, Success Rate: 0.978
    Update 2430, Success Rate: 0.971
    Update 2440, Success Rate: 0.990
    Update 2450, Success Rate: 0.981
    Update 2460, Success Rate: 0.998
    Update 2470, Success Rate: 1.000
    Update 2480, Success Rate: 0.990
    Update 2490, Success Rate: 0.997
    Update 2500, Success Rate: 0.975
    Final Evaluation Safe Rate: 0.9887`

    """