import torch

# When the actor/policy makes a decision on which direction to apply a force,
# it needs more information that just its current position and velocity in order to follow the track
# To help the actor/policy make a more "informed" decision,
# we get the `TRAJECTORY_LOOK_AHEAD_COUNT` closest points of the expert
# trajectory to the current location of the actor/policy.
# We then append these points to the current state, normalize it by `STATE_SCALE`,
# and feed it to the actor/policy.
TRAJECTORY_LOOK_AHEAD_COUNT = 10
# Dimension of the states of a trajectory. (pos_x, pos_y, velocity_x, velocity_y)
STATE_DIM = 4
# Dimension of the action a policy/actor takes. (acceleration_x, acceleration_y)
ACTION_DIM = 2
# NOTE: Tensors of size (batch_size, STATE_LOOK_AHEAD_DIM) must be accepted as input to the
# policy model
STATE_LOOK_AHEAD_DIM = STATE_DIM * (TRAJECTORY_LOOK_AHEAD_COUNT + 1)
# Scale of the states. We can use this value to scale from a normalized representation of the state
# to the true representation (or vice-versa).
STATE_SCALE = 20
# Scale of the actions. We can use this value to scale from a normalized representation of the actions
# to the true representation (or vice-versa).
ACTION_SCALE = 35


def get_closest_point(state, expert_sample):
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


def get_state_with_look_ahead_normalized(state, expert_sample):
    """
    :param state: The current state (pos_x, pos_y, v_x, v_y) of the actor
    :return: Finds the closest state to the actor on the expert trajectory and gets the next
    `TRAJECTORY_LOOK_AHEAD_COUNT` states after it. We then concatenate this with the current state
    and normalize based on `STATE_SCALE`.

    """
    expert_sample_indices_doubled = torch.arange(expert_sample.shape[0] * 2).to(expert_sample.device)
    if len(state.shape) == 1:
        state = state.unsqueeze(0)
    _, _, index = get_closest_point(state, expert_sample)
    if len(index.shape) == 0:
        index = index.unsqueeze(0)
    mask = expert_sample_indices_doubled.repeat(state.shape[0], 1)
    mask = (index.unsqueeze(-1) <= mask) & (
            mask < index.unsqueeze(-1) + TRAJECTORY_LOOK_AHEAD_COUNT)
    look_ahead = expert_sample.unsqueeze(0).repeat(state.shape[0], 2, 1)[mask]
    if len(look_ahead.shape) == 2:
        look_ahead = look_ahead.unsqueeze(0)
    look_ahead = look_ahead.reshape(state.shape[0], STATE_LOOK_AHEAD_DIM - 4)
    return torch.cat([state, look_ahead], dim=-1) / STATE_SCALE


class Critic:
    def __init__(self, model_path, device):
        self.model = torch.jit.load(model_path, map_location=device)

    def criticize(self, normalized_look_ahead_states, normalized_actions):
        """
        :param normalized_look_ahead_states: Accepts a tensor of dimension (T, STATE_LOOK_AHEAD_DIM)
        This is supposed to be a list of T timesteps of normalized_look_ahead_states that the policy
        model takes during a rollout.
        :param normalized_actions: Accepts a tensor of dimension (T, ACTION_DIM)
        This is supposed to be a list of T timesteps of normalized_actions that the policy model
        takes during a rollout.
        :return: A single (back-propable) loss for the policy model
        """
        return torch.mean(self.model(normalized_look_ahead_states, normalized_actions))


class Actor:

    def __init__(self, model, device, expert_sample):
        self.model = model
        self.model.to(device)
        self.expert_sample = expert_sample

    def act(self, state):
        """
        :param state: The state of the system
        :return: (normalized_action, state_with_look_ahead)
        - The `normalized_action` is the raw output of the policy model.
        - The `state_with_look_ahead` is the normalized state of the actor appended with
        `TRAJECTORY_LOOK_AHEAD_COUNT` closest states of the expert trajectory to the current state.
        NOTE: `(List[normalized_look_ahead_state], List[normalized_action])` is the expected input
        to the critic model.
        NOTE: To get the true action of the actor,
        multiply the normalized action with `ACTION_SCALE`:
        `action = normalized_action * ACTION_SCALE`
        """
        state_with_look_ahead = get_state_with_look_ahead_normalized(state, self.expert_sample)
        normalized_action = self.model(state_with_look_ahead)
        return normalized_action, state_with_look_ahead
