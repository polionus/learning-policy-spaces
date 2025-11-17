import numpy as np
import torch

# from karel.world import World
from tasks import Task
from vae.models import StrimmedVAE

###TODO: Clean up this class so that it works faster. 
class Policy:
    # This class recevies a tensor, and will execute the policy for us
    def __init__(self, z: torch.Tensor, conditional_model: StrimmedVAE, task_env: Task):
        self.latent_vec = z
        # We need the network that will execute the policy
        self.model = conditional_model
        self.task_env = task_env
        self.demos_per_program = 1
        self.batch_size = 1

        ones = torch.ones(
            (self.batch_size * self.demos_per_program, 1),
            dtype=torch.long,
            device=self.model.device,
        )
        self.prev_action = (self.model.num_agent_actions - 1) * ones
        self.terminated_policy = torch.zeros_like(
            self.prev_action, dtype=torch.bool, device=self.model.device
        )

    def execute_policy(self, state):

        # TODO: check terminated policy

        self.model.eval()
        z = self.latent_vec
        # Taking only first state and squeezing over first 2 dimensions
        demos_per_program = 1
        batch_size = 1

        # Need Tensor :(
        # states_np = states.detach().cpu().numpy().astype(np.bool_)
        # C x H x W to H x W x C
        # states_np = np.moveaxis(states_np, [-1, -2, -3], [-2, -3, -1])
        states_np = np.moveaxis(state.s.copy(), [-2, -3, -1], [-1, -2, -3])
        current_state = torch.from_numpy(states_np).float()

        # current_state = current_state[:, :, 0, :, :, :].view(
        #     self.batch_size * self.demos_per_program, c, h, w
        # )
        current_state = current_state.unsqueeze(0)

        # Pad the state representation:

        # We are keeping track of the current action, and the first time the policy is being
        # executed, the right initialization will occur in the program.
        # current_action = (self.model.num_agent_actions - 1) * ones
        current_action = self.prev_action

        z_repeated = z.unsqueeze(1).repeat(1, demos_per_program, 1)
        z_repeated = z_repeated.view(
            batch_size * demos_per_program, self.model.hidden_size
        )

        gru_hidden = z_repeated.unsqueeze(0)

        terminated_policy = self.terminated_policy

        mask_valid_actions = torch.tensor(
            (self.model.num_agent_actions - 1) * [-torch.finfo(torch.float32).max]
            + [0.0],
            device=self.model.device,
        )

        enc_state = self.model.state_encoder(current_state)

        enc_action = self.model.action_encoder(current_action.squeeze(-1))

        gru_inputs = torch.cat((z_repeated, enc_state, enc_action), dim=-1)
        gru_inputs = gru_inputs.unsqueeze(0)

        gru_out, gru_hidden = self.model.policy_gru(gru_inputs, gru_hidden)
        gru_out = gru_out.squeeze(0)

        pred_action_logits = self.model.policy_mlp(gru_out)

        masked_action_logits = (
            pred_action_logits + terminated_policy * mask_valid_actions
        )

        current_action = (
            self.model.softmax(masked_action_logits).argmax(dim=-1).view(-1, 1)
        )

        # Update the state of the network to keep track of the last action.
        self.prev_action = current_action

        self.terminated_policy = torch.logical_or(
            current_action == self.model.num_agent_actions - 1, terminated_policy
        )

        # Execute current action, and get the reward from the task

        # Run the action in the world
        if current_action.item() != 5:
            state.run_action(current_action)

        # Now the task should get the reward
        # return terminated, reward
        # Important Note: Oh god.
        return terminated_policy.item()
