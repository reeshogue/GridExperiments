#I haven't programmed this in a while so...

import torch
import collections
import random
import numpy as np
import copy 
import matplotlib.pyplot as plt


class EmptyUnmotivatedGrid:
    def __init__(self, size=(1,1,32,32)):
        self.size = size[-2], size[-1]
        self.grid = torch.zeros(size)
        self.player = [random.randint(0, size[-2]), random.randint(0, size[-1])]
    def step(self, action):
        try:
            if action == 0:
                self.player[0] += 1
            elif action == 1:
                self.player[0] -= 1
            elif action == 2:
                self.player[1] += 1
            elif action == 3:
                self.player[1] -= 1
            elif action == 4:
                pass
            self.grid[0][0][self.player[0]][self.player[1]] += 1
        except:
            pass

        return self.grid, self.player

class ActorNet(torch.nn.Module):
    def __init__(self, size=4):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 12, (3,3))
        self.conv2 = torch.nn.Conv2d(12, 12, (3,3))
        self.conv3 = torch.nn.Conv2d(12, 1, (3,3))
        self.linear = torch.nn.Linear(26*26, 5)
    def forward(self, x, adversary, timestep):
        x = self.conv(x + adversary)
        x += timestep
        x = torch.sigmoid(x) * x
        x = self.conv2(x)
        x = torch.sigmoid(x) * x
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = torch.softmax(x, dim=1)
        return x

class QNet(torch.nn.Module):
    def __init__(self, size=4):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 12, (3,3))
        self.conv2 = torch.nn.Conv2d(12, 12, (3,3))
        self.conv3 = torch.nn.Conv2d(12, 1, (3,3))
        self.linear = torch.nn.Linear(26*26, 6)
    def forward(self, x, adversary, timestep):
        x = self.conv(x + adversary)
        x += timestep
        x = torch.sigmoid(x) * x
        x = self.conv2(x)
        x = torch.sigmoid(x) * x
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x

class AdversaryNet(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 12, (3,3))
        self.conv2 = torch.nn.Conv2d(12, 1, (3,3))
        self.linear = torch.nn.Linear(28*28, 1)

        self.tconv = torch.nn.ConvTranspose2d(12, 1, (3,3))
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)
        o_x = self.conv(x)
        x = self.conv2(o_x)
        x = torch.flatten(x, start_dim=1)
        timesteps = self.linear(x)
        x = o_x


        x = torch.sigmoid(x) * x
        x = self.tconv(x)
        return x, timesteps

def kl_div(x1, x2):
    return torch.sum(x1*(torch.log(x1/x2)))

def jensen_shannon(input, target):
    m = .5 * (input + target)
    y = .5 * kl_div(input, m) + .5 * kl_div(target, m)
    return y

class Agent:
    def __init__(self, size):

        self.actor = ActorNet(size)
        self.optim_actor = torch.optim.AdamW(self.actor.parameters(), lr=1e-3)

        self.adversary = AdversaryNet(size)
        self.optim_advers = torch.optim.AdamW(self.adversary.parameters(), lr=1e-3)

        self.adversary_q = QNet(size)
        self.optim_q = torch.optim.AdamW(self.adversary_q.parameters(), lr=1e-3)

        self.memory = collections.deque(maxlen=100)
    def act(self, grid):
        return self.actor(grid, *self.adversary(grid))
    def remember(self, grid, action, timestep):
        self.memory.append((grid, action, timestep))
    def replay(self):
        random_a = random.sample(self.memory, 1)
        random_b = random.sample(self.memory, 1)
        if random_a[-1][-1] > random_b[-1][-1]:
            random_b, random_a = random_a, random_b
        for (grid, action, timestep), (gridn, actionn, timestepn) in zip(random_a, random_b):
            with torch.no_grad():
                grid, action, timestep = grid, action, timestep
                gridn, actionn, timestepn = gridn, actionn, timestepn
            delta_timesteps = timestepn - timestep

            self.optim_actor.zero_grad()
            faction = self.actor(grid, grid+gridn, delta_timesteps)
            aaction = self.actor(grid, *self.adversary(grid))
            loss_action = jensen_shannon(faction, action)
            loss = jensen_shannon(aaction, action) + loss_action
            loss.backward()
            self.optim_actor.step()

            self.optim_advers.zero_grad()
            faction = self.actor(grid, *self.adversary(grid))
            loss_adversary = -torch.mean(self.adversary_q(grid, *self.adversary(grid)))
            loss_adversary.backward()
            self.optim_advers.step()

            self.optim_q.zero_grad()
            loss_action = loss_action.detach()
            floss_action = self.adversary_q(grid, *self.adversary(grid))
            floss_action = torch.mean(floss_action)
            loss_q = torch.nn.functional.mse_loss(floss_action, loss_action)
            loss_q.backward()
            self.optim_q.step()


def show_grid(grid, player):
    grid = grid.clone().detach()
    grid = torch.sigmoid(grid)
    grid[0][0][player[0]][player[1]] = 2
    grid = grid.squeeze(0).squeeze(0).numpy()
    plt.imshow(grid)
    plt.pause(0.01)
    plt.show(block=False)


def main():
    env = EmptyUnmotivatedGrid()
    agent = Agent((32,32))
    grid = env.grid
    timestep = torch.tensor(0.0)
    while True:
        with torch.no_grad():
            action = agent.act(grid)
            action_argmax = np.random.choice(5, p=action[0].numpy())
            grid, player = env.step(action_argmax)
            show_grid(grid, player)
        agent.remember(grid, action, timestep)
        agent.replay()
        timestep += 1.0     
if __name__ == "__main__":
    main()
