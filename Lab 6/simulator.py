import numpy as np
import torch


Ts = 0.05
drag_coefficient = 0.1

d_max = 1
v_max = 20
a_max = 1

c = 0.075


def dynamics_ca(dt: float):
    dt = Ts
    A = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    B = np.array(
        [
            [dt**2 / 2, 0.0],
            [0.0, dt**2 / 2],
            [dt, 0.0],
            [0.0, dt],
        ]
    )
    return A, B


def dynamics_ca_drag(dt: float, mu: float):
    A, B = dynamics_ca(dt)
    A[0, 2] -= mu * dt**2 / 2
    A[1, 3] -= mu * dt**2 / 2
    A[2, 2] -= mu * dt
    A[3, 3] -= mu * dt
    return A, B


class Car:
    def __init__(self) -> None:
        self.A, self.B = dynamics_ca_drag(Ts, drag_coefficient)
        ref = np.load('lerp_expert_sample.npy')
        self.ref = torch.from_numpy(ref).double()
        self.rng = np.random.default_rng()

    def get_ref_distance(self, s):
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        idx = torch.argmin(torch.sqrt(torch.sum((self.ref.unsqueeze(0) - s.unsqueeze(1)) ** 2, dim=-1)), dim=-1)
        s_ref = self.ref[idx]
        d = torch.sqrt(torch.sum((s_ref - s) ** 2, dim=-1))
        return d

    def get_reward(self, s, a):
        max_reward = 1.5
        d = self.get_ref_distance(s)
        if d > d_max:
            return torch.tensor([-1])
        return max_reward - d - c * torch.sqrt(torch.sum(a) ** 2)

    def reset(self):
        return self.ref[int(torch.rand(1).item() * self.ref.shape[0])].unsqueeze(0)

    def step(self, s, a):
        r = self.get_reward(s.detach(), a.detach())
        s_next = s.detach() @ self.A.T + a.detach() @ self.B.T

        d = self.get_ref_distance(s_next) 
        done = d > d_max

        return s_next, r, done
