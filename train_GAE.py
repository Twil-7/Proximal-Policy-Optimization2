import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical  # 提供了一组概率分布的实现，这里用于表示离散类别型分布

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
        self.y = INITIAL_HEIGHT  # 10.0
        self.x = np.random.uniform(X_BOUNDS[0], X_BOUNDS[1])  # (0.0, 10.0)
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
            if TARGET_RANGE[0] <= self.x <= TARGET_RANGE[1]:
                return 10.0
            return -10.0

        distance = abs(self.x - np.mean(TARGET_RANGE))
        in_target_air = TARGET_RANGE[0] <= self.x <= TARGET_RANGE[1]

        reward = 0.0
        reward += 0.2 if in_target_air else -0.1
        reward -= 0.05 * distance
        reward -= 0.1

        return reward


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        # print(state_dim, action_dim, hidden_size): 2 2 128
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),  # (2, 128)
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # (128, 128)
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),  # (128, 2)
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        # print(state_dim, hidden_size): 2 128
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),  # (2, 128)
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # (128, 128)
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # (128, 1)
        )

    def forward(self, x):
        return self.net(x)


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2,
                 gae_lambda=0.95, epochs=4, batch_size=64, critic_lr=3e-4):
        # print(state_dim, action_dim): 2 2    # 状态空间用二维的坐标表示，动作空间用二维的动作概率表示，向左或向右一格的概率
        # print(lr): 0.0003    # 学习率
        # print(gamma): 0.99    # 时序差分的折扣
        # print(epsilon): 0.2    # 用在clip函数限制梯度更新幅度
        # print(gae_lambda): 0.95    # 广义优势函数中的指数加权平均
        # print(epochs): 4    # 一批旧的轨迹样本更新4次参数
        # print(batch_size): 64    # 每次采样64个状态动作对

        self.actor = Actor(state_dim, action_dim).to(device)  # (2, 2)
        self.critic = Critic(state_dim).to(device)  # 2
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)  # 0.0003
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)  # 0.0003
        self.gamma = gamma  # 0.99
        self.epsilon = epsilon  # 0.2
        self.gae_lambda = gae_lambda  # 0.95
        self.epochs = epochs  # 4
        self.batch_size = batch_size  # 64

    def update(self, episodes):
        states, actions, old_probs, advantages, v_targets = [], [], [], [], []

        # 处理每个episode的数据
        for episode in episodes:
            states_ep = episode['states']
            next_states_ep = episode['next_states']
            rewards_ep = episode['rewards']
            dones_ep = episode['dones']
            actions_ep = episode['actions']
            old_probs_ep = episode['old_probs']
            # print(states_ep): [array([0.9507143, 1.       ]), ... , array([0.6507143, 0.1      ]]
            # print(next_states_ep): [array([0.8507143, 0.9      ]), ... ,  array([0.5507143, 0.       ])]
            # print(rewards_ep): [-0.375357153204958, -0.32535715320495806, ... , , -0.275357153204958, 10.0]
            # print(dones_ep): [False, False, False, False, False, False, False, False, False, True]
            # print(actions_ep): [0, 0, 0, 0, 1, 0, 1, 0, 1, 0]
            # print(old_probs_ep): [0.5160935521125793, 0.514032781124115, ... ,0.5018236041069031, 0.4957466125488281]

            # 转换为tensor
            states_tensor = torch.tensor(states_ep, dtype=torch.float32).to(device)
            next_states_tensor = torch.tensor(next_states_ep, dtype=torch.float32).to(device)
            rewards_tensor = torch.tensor(rewards_ep, dtype=torch.float32).to(device)
            dones_tensor = torch.tensor(dones_ep, dtype=torch.bool).to(device)
            # print(states_tensor.shape): torch.Size([10, 2])
            # print(next_states_tensor.shape): torch.Size([10, 2])
            # print(rewards_tensor.shape): torch.Size([10])
            # print(dones_tensor.shape): torch.Size([10])

            """  无论是（蒙特卡洛）MC方法, （时序差分）TD方法, 还是（广义优势函数）GAE方法，Advantage计算的本质就是: 
            A(at, st) = Q(at, st) - V(st)，这里V(st)都是通过Critic网络估计的，然后将Q(at, st)作为V(st)的Ground Truth用来监督，
            从而更新Critic网络参数。但不同之处在于:
            (1) （蒙特卡洛）MC方法，直接定义Q(at, st)为当前时间步的累积折扣汇报，在此基础上计算A(at, st)；
            (2) （时序差分）TD方法，对Q(at, st)进一步改进，将其定义为当前步奖励+gamma*下一状态值，同样在此基础上计算A(at, st)；
            (3) （广义优势函数）GAE方法，没有显式构造出Q(at, st)的表达式，而是直接构造得到A(at, st)的表达式，然后通过
            Q(at, st)=A(at, st) + V(st)，将计算得到的Q(at, st)作为Critic网络的监督信息；
            """

            with torch.no_grad():
                V = self.critic(states_tensor).squeeze()
                V_next = self.critic(next_states_tensor).squeeze()
                V_next[dones_tensor] = 0.0  # 终止状态后的V值为0
                # print(V): tensor([0.1777, 0.1765, 0.1776, 0.1777, 0.1759, 0.1745, 0.1715, 0.1614, 0.1542, 0.1356])
                # print(V_next): tensor([0.1765, 0.1776, 0.1777, 0.1759, 0.1745, 0.1715, 0.1614, 0.1542, 0.1356, 0.0])

            # 计算TD残差
            deltas = rewards_tensor + self.gamma * V_next - V  # 一步时序差分
            # print(deltas): tensor([-0.3784, -0.3260, -0.2771,  0.0711, -0.2784,  0.0699, -0.2870,  0.0659,
            # -0.2954,  9.8644], device='cuda:0')

            # 计算GAE
            advantages_ep = []
            gae = 0
            for delta, done in zip(reversed(deltas.cpu().numpy()), reversed(dones_ep)):
                if done:  # 终止状态后的gae值为0
                    gae = 0
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages_ep.insert(0, gae)
            # print(advantages_ep): [4.305473888460532, 4.980173877112112, 5.641845985898013, 6.293382006091043,
            # 6.615925525155157, 7.330531840463924, 7.719972048861672, 8.513567713694572, 8.982118234634399, 9.8644447]

            advantages_ep = torch.tensor(advantages_ep, dtype=torch.float32).to(device)
            # print(advantages_ep): tensor([4.3055, 4.9802, 5.6418, 6.2934, 6.6159, 7.3305, 7.7200, 8.5136, 8.9821,
            #         9.8644], device='cuda:0')

            # 标准化优势值（按episode）
            advantages_ep = (advantages_ep - advantages_ep.mean()) / (advantages_ep.std() + 1e-8)
            # print(advantages_ep): tensor([-1.5181, -1.1414, -0.7720, -0.4083, -0.2282,  0.1707,  0.3881,  0.8312,
            #          1.0927,  1.5853], device='cuda:0')

            # 计算V目标值
            v_target_ep = advantages_ep + V.detach()  # 用于监督Critic网络输出
            # print(v_target_ep): tensor([-1.3404, -0.9650, -0.5944, -0.2306, -0.0523,  0.3453,  0.5596,  0.9926,
            #          1.2470,  1.7209], device='cuda:0')

            # 收集数据
            states.extend(states_ep)
            actions.extend(actions_ep)
            old_probs.extend(old_probs_ep)
            advantages.extend(advantages_ep.cpu().numpy().tolist())
            v_targets.extend(v_target_ep.cpu().numpy().tolist())

        # 转换为tensor
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
        old_probs_tensor = torch.tensor(old_probs, dtype=torch.float32).to(device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
        v_targets_tensor = torch.tensor(v_targets, dtype=torch.float32).to(device)
        # print(states_tensor.shape): torch.Size([100, 2])
        # print(actions_tensor.shape): torch.Size([100])
        # print(old_probs_tensor.shape): torch.Size([100])
        # print(advantages_tensor.shape): torch.Size([100])
        # print(v_targets_tensor.shape): torch.Size([100])

        # 训练多个epoch
        for _ in range(self.epochs):  # 4
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]  # 从中随机筛选出64个状态动作对

                batch_states = states_tensor[idx]
                batch_actions = actions_tensor[idx]
                batch_old_probs = old_probs_tensor[idx]
                batch_advantages = advantages_tensor[idx]
                batch_v_targets = v_targets_tensor[idx]
                # print(batch_states.shape): torch.Size([64, 2])
                # print(batch_actions.shape): torch.Size([64])
                # print(batch_old_probs.shape): torch.Size([64])
                # print(batch_advantages.shape): torch.Size([64])
                # print(batch_v_targets.shape): torch.Size([64])

                # 更新Actor
                new_probs = self.actor(batch_states)
                # print(new_probs.shape): torch.Size([64, 2])
                new_probs = new_probs.gather(1, batch_actions.unsqueeze(1)).squeeze()  # 计算新策略，执行同样动作的概率
                # print(new_probs.shape): torch.Size([64])

                ratio = new_probs / batch_old_probs
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                # print(ratio.shape): torch.Size([64])
                # print(surr1.shape): torch.Size([64])
                # print(surr2.shape): torch.Size([64])
                # print(actor_loss): tensor(-0.0989, device='cuda:0', grad_fn=<NegBackward0>)

                # 更新Critic
                v_pred = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(v_pred, batch_v_targets)
                # print(v_pred.shape): torch.Size([64])
                # print(critic_loss): tensor(0.8529, device='cuda:0', grad_fn=<MseLossBackward0>)

                """
                我匪夷所思难以理解的是：
                1、为什么可以通过这种方式来监督Critic网络，并构造出critic_loss，明明没有Ground Truth的状态值用于监督？
                2、在更新Critic网络时，也是用一批旧的轨迹多次更新网络参数，会存在影响吗？
                3、可以不使用critic_loss，而仅仅使用actor_loss吗，毕竟计算actor_loss时也会与状态值估计有关？

                我目前有一个理解是，Actor网络和Critic网络其实采用的是交叉迭代、不断优化的思想：在更新Actor网络参数的时候，先假设Critic网络
                估计出的状态值都是准确的，从而构造出状态动作对的优势值是准确的，由此构造出actor_loss；在更新Critic网络参数的时候，
                假设Actor网络更新的决策是准确的，状态动作对的优势也是准确的，这时候相邻状态之间就应该是这个优势，从而构建出critic_loss。
                """

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


def collect_episode_data(env, actor):
    episode = {'states': [], 'next_states': [], 'actions': [],
               'old_probs': [], 'rewards': [], 'dones': []}
    state = env.reset()
    # print(state): [0.9507143 1.       ]
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).to(device)
        # print(state_tensor): tensor([0.9507, 1.0000], device='cuda:0')
        with torch.no_grad():
            action_probs = actor(state_tensor)
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

        episode['states'].append(state)
        episode['next_states'].append(next_state)
        episode['actions'].append(action.item())
        episode['old_probs'].append(old_prob)
        episode['rewards'].append(reward)
        episode['dones'].append(done)

        state = next_state

    return episode


def train():
    env = DropBlockEnv()
    state_dim = 2
    action_dim = 2
    ppo = PPO(state_dim, action_dim)  # (2, 2)

    num_episodes_per_update = 10    # 每次更新时，采样10条轨迹数据
    total_updates = 2501    # 总共进行2501次更新（每次更新时，会基于旧的轨迹数据，对参数优化更新4次）
    eval_interval = 10

    success_rate = evaluate(ppo, num_episodes=10000)
    print(f"Initial Evaluation Safe Rate: {success_rate:.4f}")

    for update in range(total_updates):  # 执行2501次，每次采样10条轨迹数据，并使用这批数据更新4次参数
        # 收集数据
        episodes = []
        for _ in range(num_episodes_per_update):  # 10    # 采样10条轨迹
            episodes.append(collect_episode_data(env, ppo.actor))

        # 更新策略
        ppo.update(episodes)

        # 评估性能
        if (update + 1) % eval_interval == 0:
            success_rate = evaluate(ppo, num_episodes=1000)
            print(f"Update {update + 1}, Safe Rate: {success_rate:.4f}")

    # 最终评估
    success_rate = evaluate(ppo, num_episodes=10000)
    print(f"Final Evaluation Safe Rate: {success_rate:.4f}")
    torch.save(ppo.actor.state_dict(), 'policy_GAE.pth')


if __name__ == "__main__":
    train()

    """
        Initial Evaluation Safe Rate: 0.1484
        Update 10, Safe Rate: 0.1250
        Update 20, Safe Rate: 0.5890
        Update 30, Safe Rate: 0.8230
        Update 40, Safe Rate: 0.7480
        Update 50, Safe Rate: 0.9200
        Update 60, Safe Rate: 0.9170
        Update 70, Safe Rate: 0.9970
        Update 80, Safe Rate: 0.9510
        Update 90, Safe Rate: 0.9680
        Update 100, Safe Rate: 0.9930
        ......
        Update 2400, Safe Rate: 0.9950
        Update 2410, Safe Rate: 0.9800
        Update 2420, Safe Rate: 0.9860
        Update 2430, Safe Rate: 0.9990
        Update 2440, Safe Rate: 0.9930
        Update 2450, Safe Rate: 0.9800
        Update 2460, Safe Rate: 0.9780
        Update 2470, Safe Rate: 0.9870
        Update 2480, Safe Rate: 0.9890
        Update 2490, Safe Rate: 0.9480
        Update 2500, Safe Rate: 0.9660
        Final Evaluation Safe Rate: 0.9855
    
    """
