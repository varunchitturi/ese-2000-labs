import torch


state_scale = 20
action_scale = 35

class Critic:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path).double()

    def evaluate(self, s, obs, a):
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        s_aug = torch.cat([s, obs], dim=-1) / state_scale
        a = a / action_scale
        return self.model(s_aug, a)
