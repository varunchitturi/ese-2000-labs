import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ese2000_dynamical.simulator import Simulator



class Critic:
    STATE_SCALE = 20
    ACTION_SCALE = 35
    TRAJECTORY_INDEX = 5
    TRAJECTORY_LOOK_AHEAD_COUNT = 10
    STATE_DIM = 4
    ACTION_DIM = 2
    STATE_LOOK_AHEAD_DIM = STATE_DIM * (TRAJECTORY_LOOK_AHEAD_COUNT + 1)

    def __init__(self,
                 model_path,
                 expert_trajectory_path="./ese2000-dynamical-systems/data",
                 device="cpu",
                 trajectory_interpolation_amount=2):
        simulator = Simulator()
        self.sim_weight_A = torch.Tensor(simulator.A).to(device)
        self.sim_weight_B = torch.Tensor(simulator.B).to(device)
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = torch.jit.load(model_path, map_location=device).to(device)
        self.trajectory_interpolation_amount = trajectory_interpolation_amount
        self.expert_trajectories = np.load(f"{expert_trajectory_path}/states.npy")
        self.expert_sample = Critic._interpolate_trajectory(
            torch.tensor(self.expert_trajectories[Critic.TRAJECTORY_INDEX])
            .float()
            .to(device),
            trajectory_interpolation_amount
        )
        print(self.expert_sample.shape)
        self.expert_sample_indices_doubled = torch.arange(self.expert_sample.shape[0] * 2).to(self.device)
        print(
            f"The normalized_state_look_ahead dim (input dim for policy) is {Critic.STATE_LOOK_AHEAD_DIM}")

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
        `Critic.TRAJECTORY_LOOK_AHEAD_COUNT` states after it. We then concatenate this with the current state
        and normalize based on `STATE_SCALE`.

        """

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        _, _, index = Critic._get_closest_point(state, self.expert_sample)
        if len(index.shape) == 0:
            index = index.unsqueeze(0)
        mask = self.expert_sample_indices_doubled.repeat(state.shape[0], 1)
        mask = (index.unsqueeze(-1) <= mask) & (
                mask < index.unsqueeze(-1) + Critic.TRAJECTORY_LOOK_AHEAD_COUNT)
        look_ahead = self.expert_sample.unsqueeze(0).repeat(state.shape[0], 2, 1)[mask]
        if len(look_ahead.shape) == 2:
            look_ahead = look_ahead.unsqueeze(0)
        look_ahead = look_ahead.reshape(state.shape[0], Critic.STATE_LOOK_AHEAD_DIM - 4)
        return torch.cat([state, look_ahead], dim=-1) / Critic.STATE_SCALE

    def run_policy(self, policy_model, random_start=True, max_timesteps=250, truncate_distance=1):
        """

        :param policy_model: The policy model to evaluate. Note: The policy model must take as input
        a `(batch_size, Critic.STATE_LOOK_AHEAD_DIM)` dimension tensor.
        :param random_start: Whether the policy model should start at the first timestep of the
        expert trajectory or anywhere along the expert trajectory.
        :param max_timesteps: The maximum number of timesteps to evaluate the policy for.
        :param truncate_distance: Stops evaluating the policy when it deviates more than `truncate_distance`
        from the expert trajectory.
        :return: (Normalized states with look ahead that the policy went through,
                  The normalized actions that the policy took,
                  The trajectory that policy took)
        """
        predicted_trajectory = []
        # normalized policy states with look ahead
        policy_states = []
        # normalized policy actions
        policy_actions = []
        state = self.expert_sample[0].unsqueeze(0)
        if random_start:
            state = self.expert_sample[
                int(torch.rand(1).item() * self.expert_sample.shape[0])].unsqueeze(0)
        predicted_trajectory.append(state)
        for t in range(max_timesteps - 1):
            state_with_look_ahead_normalized = self._get_state_with_look_ahead_normalized(state)
            policy_states.append(state_with_look_ahead_normalized)

            normalized_action = policy_model(state_with_look_ahead_normalized)
            policy_actions.append(normalized_action)
            action = normalized_action * Critic.ACTION_SCALE
            next_state = state.detach() @ self.sim_weight_A.T + action.detach() @ self.sim_weight_B.T
            predicted_trajectory.append(next_state)
            if self._get_closest_point(state, self.expert_sample)[0] > truncate_distance:
                break
            state = next_state

        policy_states = torch.stack(policy_states).squeeze(1)
        policy_actions = torch.stack(policy_actions).squeeze(1)
        predicted_trajectory = torch.stack(predicted_trajectory).squeeze(1)

        return policy_states, policy_actions, predicted_trajectory

    def criticize_policy(self, policy_model, max_timesteps=250, truncate_distance=1):
        """

        :param policy_model: The policy model to evaluate. Note: The policy model must take as input
        a `(batch_size, Critic.STATE_LOOK_AHEAD_DIM)` dimension tensor.
        :param max_timesteps: The maximum number of timesteps to evaluate the policy for.
        :param truncate_distance: Stops evaluating the policy when it deviates more than `truncate_distance`
        from the expert trajectory.
        :return: (A (back-propable) loss for the policy, the trajectory that policy took).
        """

        (policy_states,
         policy_actions,
         predicted_trajectory) = self.run_policy(policy_model,
                                                 max_timesteps=max_timesteps,
                                                 truncate_distance=truncate_distance)

        critic_evaluation = torch.mean(self.model(policy_states, policy_actions))
        return critic_evaluation, predicted_trajectory.detach()
