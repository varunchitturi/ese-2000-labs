import torch
import numpy as np
from ese2000_dynamical.simulator import Simulator

class Critic:
    STATE_SCALE = 20
    ACTION_SCALE = 35
    TRAJECTORY_INDEX = 5

    def __init__(self,
                 model_path,
                 expert_trajectory_path,
                 device="cpu",
                 trajectory_look_ahead_count=10,
                 trajectory_interpolation_amount=2):
        self.physics_simulator = Simulator()
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = torch.load(model_path).to(device)
        self.state_dim = 4
        self.action_dim = 2
        self.look_ahead_state_dim = self.state_dim * (trajectory_look_ahead_count + 1)
        self.trajectory_interpolation_amount = trajectory_interpolation_amount
        self.expert_trajectories = np.load(f"{expert_trajectory_path}/states.npy")
        self.trajectory_look_ahead_count = trajectory_look_ahead_count
        self.expert_sample = Critic._interpolate_trajectory(
            torch.tensor(self.expert_trajectories[Critic.TRAJECTORY_INDEX])
            .float()
            .to(device),
            trajectory_interpolation_amount
        )

    @staticmethod
    def _interpolate_trajectory(trajectory, amount):
        """

        :param trajectory: The trajectory to interpolate
        :param amount: The amount of points to interpolate between each of the trajectory states
        :return: The expert trajectory with `amount` points linearly interpolated between each trajectory state
        """
        new_trajectory = []
        for i, point in enumerate(trajectory):
            average = (trajectory[(i + 1) % trajectory.shape[0]] + trajectory[i]) / 2
            new_trajectory.append(trajectory[i])
            new_trajectory.append(average)
        if amount == 1:
            return torch.stack(new_trajectory)
        return Critic._interpolate_trajectory(torch.stack(new_trajectory), amount - 1)

    @staticmethod
    def _get_closest_point(state, expert_sample):
        """

        :param state: The current state of the actor
        :param expert_sample: The expert sampled trajectory to get the closest point from
        :return: The closest point in the expert trajectory to the current state
        """
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        index = torch.argmin(torch.sqrt(
            torch.sum((expert_sample.unsqueeze(0) - state.unsqueeze(1)) ** 2, dim=-1)), dim=-1)
        closest_point = expert_sample[index]
        distance = torch.sqrt(torch.sum((closest_point - state) ** 2, dim=-1))
        return distance, closest_point, index

    def _get_state_with_look_ahead_normalized(self, state):
        """
        :param state: The current state (pos_x, pos_y, v_x, v_y) of the actor
        :return: Finds the closest state to the actor on the expert trajectory and gets the next
        `trajectory_look_ahead` states after it. We then concatenate this with the current state
        and normalize based on `STATE_SCALE`.

        """
        all_indices = torch.arange(self.expert_sample.shape[0] * self.trajectory_interpolation_amount).unsqueeze(0)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        _, _, index = Critic._get_closest_point(state)
        if len(index.shape) == 0:
            index = index.unsqueeze(0)
        mask = all_indices.repeat(state.shape[0], 1)
        mask = (index.unsqueeze(-1) <= mask) & (mask < index.unsqueeze(-1) + self.trajectory_look_ahead_count)
        look_ahead = self.expert_sample.unsqueeze(0).repeat(state.shape[0], 2, 1)[mask]
        if len(look_ahead.shape) == 2:
            look_ahead = look_ahead.unsqueeze(0)
        look_ahead = look_ahead.reshape(state.shape[0], self.look_ahead_state_dim - 4)
        return torch.cat([state, look_ahead], dim=-1) / Critic.STATE_SCALE

    def criticize_policy(self, policy_model, max_timesteps=250, truncate_distance=1):
        predicted_trajectory = []
        state = self.expert_sample[int(torch.rand(1).item() * self.expert_sample.shape[0])]
        for t in range(max_timesteps):
            normalized_look_ahead_state = self._get_state_with_look_ahead_normalized(state)
            normalized_action = policy_model(normalized_look_ahead_state)
            action = normalized_action * Critic.ACTION_SCALE

            next_state = state.detach() @ self.physics_simulator.A.T + action.detach() @ self.physics_simulator.B.T
            predicted_trajectory.append(next_state)
            if self._get_closest_point(state, self.expert_sample)[0] > truncate_distance:
                break
            state = next_state
        predicted_trajectory = torch.stack(predicted_trajectory).squeeze(1)
        return predicted_trajectory
