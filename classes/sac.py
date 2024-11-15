import torch
import torchviz
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchrl.data import ReplayBuffer

import sys

from classes.client import Client
from classes.model import *
from classes.utils import *

class Actor(nn.Module):
    """
    Class to represent an Actor (policy, model) in the context of the SAC algorithm. Children class of nn.Module.
    """
    def __init__(self, model: EOSModel, lr: float=1e-3):
        super(Actor, self).__init__()
        self.role_type = "Actor"
        self.model = model
        self.lr = lr

    def forward(self, states, actions):
        return self.model(states, actions)

class Critic(nn.Module):
    """
    Class to represent a Critic in the context of the SAC algorithm. Children class of nn.Module.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: tuple[int]=[256, 64, 256, 64], lr: float=1e-3):
        super(Critic, self).__init__()
        self.role_type = "Critic"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        layers = []
        
        for i in range(len(hidden_dim)):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim[i]))
            else:
                layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim[-1], out_dim))

        self.sequential = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for module in self.sequential:
            if type(module) == torch.nn.Linear:
                    nn.init.zeros_(module.bias)
                    nn.init.uniform_(module.weight, -initrange, initrange)

    def forward(self, x):
        return self.sequential(x)

class QNetwork(Critic):
    """
    Class to represent a Q-network. Children class of Critic.
    """
    def __init__(self, state_dim: int, action_dim: int, out_dim: int, hidden_dim: tuple[int]=[256, 64, 256, 64], lr: float=1e-3):
        super(QNetwork, self).__init__(in_dim=(state_dim + action_dim), out_dim=out_dim, hidden_dim=hidden_dim, lr=lr)
        self.critic_type = "Q-network"
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, state, action):
        # Fill the state with zeros if it is not the same size as the input dimension
        if state.shape[0] < self.in_dim:
            state = torch.cat([state, torch.zeros(self.state_dim - state.shape[0])])
        elif state.shape[0] > self.in_dim:
            state = state[-self.in_dim:]

        x = torch.cat([state, action])
        x = super(QNetwork, self).forward(x)
        return x
    
class VNetwork(Critic):
    """
    Class to represent a V-network. Children class of Critic.
    """
    def __init__(self, state_dim, out_dim, hidden_dim: tuple[int]=[256, 64, 256, 64], lr: float=1e-3):
        super(VNetwork, self).__init__(in_dim=state_dim, out_dim=out_dim, hidden_dim=hidden_dim, lr=lr)
        self.critic_type = "V-network"

    def forward(self, state):
        # Fill the state with zeros if it is not the same size as the input dimension
        if state.shape[0] < self.in_dim:
            state = torch.cat([state, torch.zeros(self.in_dim - state.shape[0])])
        elif state.shape[0] > self.in_dim:
            state = state[-self.in_dim:]

        x = state
        x = super(VNetwork, self).forward(x)
        return x

class SoftActorCritic():
    """
    Class to represent the Soft Actor-Critic algorithm. Children class of nn.Module.
    """
    def __init__(self, conf: DataFromJSON, client: Client):
        self.__role_type = "Soft Actor-Critic"
        self.__conf = conf
        self.client = client
        self.set_properties(conf)

    def __str__(self) -> str:
        return f"{self.__role_type} object with configuration: {self.__conf}"

    def set_properties(self, conf: DataFromJSON):
       for key, value in conf.__dict__.items():
            if not key.startswith("__"):
                setattr(self, key, value)

    def train(self):
        torch.autograd.set_detect_anomaly(True)

        # Create the embedder object for states
        states_embedder = FloatEmbedder(
            input_dim=self.state_dim,
            embed_dim=self.d_model
        )
        
        # Create the embedder object for actions
        actions_embedder = FloatEmbedder(
            input_dim=self.action_dim,
            embed_dim=self.d_model
        )
        
        # Create the positional encoder object
        pos_encoder = PositionalEncoder(
            d_model=self.d_model,
            max_len=self.max_len,
            dropout=self.pos_dropout
        )

        # Create the transformer model
        transformer = EOSTransformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout,
            activation=self.activation,
            batch_first=self.batch_first
        )
        
        # Create a linear outside stochastic layer called projector
        stochastic_projector = StochasticProjector(
            d_model=self.d_model,
            action_dim=self.action_dim
        )
        
        # Create the model object
        model = EOSModel(
            state_embedder=states_embedder,
            action_embedder=actions_embedder,
            pos_encoder=pos_encoder,
            transformer=transformer,
            projector=stochastic_projector
        )

        # Create the actor
        actor = Actor(model)

        # Create the NNs for the Q-networks
        q1 = QNetwork((self.state_dim + self.action_dim) * self.max_len, self.action_dim, 1)
        q2 = QNetwork((self.state_dim + self.action_dim) * self.max_len, self.action_dim, 1)

        # Create the NNs for the V-networks
        v = VNetwork((self.state_dim + self.action_dim) * self.max_len, 1)
        vtg = VNetwork((self.state_dim + self.action_dim) * self.max_len, 1)

        # Sending data to get the initial state
        sending_data = {
            "agent_id": 0,
            "action": {
                "d_az": 0,
                "d_el": 0
            },
            "delta_time": 0
        }
        
        state, _, _ = self.client.get_next_state("get_next", sending_data)

        # Input tensor of 1 batch and 1 sequence of state_dim dimensional states
        states = torch.FloatTensor([[state]])

        # Input tensor of 1 batch and 1 sequence of action_dim dimensional actions (equal to 0)
        actions = torch.FloatTensor([[[0 for _ in range(self.action_dim)]]])

        # Optimizers
        optimizer_v = optim.Adam(v.parameters(), lr=v.lr)
        optimizer_q1 = optim.Adam(q1.parameters(), lr=q1.lr)
        optimizer_q2 = optim.Adam(q2.parameters(), lr=q2.lr)
        optimizer_pi = optim.Adam(actor.model.parameters(), lr=actor.lr)

        # Create the replay buffer
        replay_buffer = ReplayBuffer()

        # Create the tensor manager
        tensor_manager = TensorManager()

        # Loop flags
        done = False
        iteration = 1

        print("Starting looped training...")

        # Loop over all iterations
        while not done:
            print(f"\nStarting iteration {iteration}...")
            iteration += 1

            # Loop over all environment steps
            for e in range(self.environment_steps):
                with torch.no_grad():
                    # Adjust the maximum length of the states and actions
                    states = states[:, -self.max_len:, :]
                    actions = actions[:, -self.max_len:, :]

                    # Create the augmented state
                    aug_state = [states.clone(), actions.clone()]

                    # Get the stochastic actions
                    stochastic_actions = actor(states, actions)

                    # Select the last stochastic action
                    a_sto = stochastic_actions[-1, -1, :]

                    # Sample and convert the action
                    a_pretanh, a = actor.model.reparametrization_trick(a_sto)

                    # --------------- Environment's job to provide info ---------------
                    sending_data = {
                        "agent_id": 0,
                        "action": {
                            "d_az": a[0].item() * 180,
                            "d_el": a[1].item() * 90
                        },
                        "delta_time": self.time_increment
                    }
                    
                    state, reward, done = self.client.get_next_state("get_next", sending_data)

                    # Break if time is up
                    if done:
                        print("Time is up!")
                        break

                    # Get the reward
                    r = torch.FloatTensor([reward])

                    # Get the next state
                    s_next = torch.FloatTensor(state)
                    # --------------- Environment's job to provide info ---------------

                    # Add it to the states
                    states = torch.cat([states, s_next.unsqueeze(0).unsqueeze(0)], dim=1)

                    # Add it to the actions
                    actions = torch.cat([actions, a.unsqueeze(0).unsqueeze(0)], dim=1)

                    # Adjust the maximum length of the states and actions
                    states = states[:, -self.max_len:, :]
                    actions = actions[:, -self.max_len:, :]

                    # Augmented state for the next step
                    aug_state_next = [states, actions]

                    # Store in the buffer
                    replay_buffer.add((aug_state, a, r, aug_state_next))

                if not e == 0:
                    sys.stdout.write("\033[F")
                print(f"Environment step {e+1}/{self.environment_steps} done!")
            
            # Break if time is up
            if done:
                break

            # Loop over all gradient steps
            for g in range(self.gradient_steps):
                with torch.no_grad():
                    aug_state, a, r, aug_state_next = tensor_manager.full_squeeze(*replay_buffer.sample(1))

                # Batchify the tensors neccessary for the transformer
                aug_state, aug_state_next = tensor_manager.batchify(aug_state, aug_state_next)

                # Get the stochastic actions again
                new_stochastic_actions = actor(aug_state[0], aug_state[1])

                # Select the last stochastic action
                a_new_sto = new_stochastic_actions[-1, -1, :]

                # Sample and convert the action
                a_new_pretanh, a_new = actor.model.reparametrization_trick(a_new_sto)

                # Reshape the augmented state to input to the Q-networks and the V-network as a single 1D tensor
                aug_state_1D = torch.cat([aug_state[0], aug_state[1]], dim=2).view(-1)

                # Find the minimum of the Q-networks
                qmin = torch.min(q1(aug_state_1D, a_new), q2(aug_state_1D, a_new))

                # Log probability
                sum = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
                corrective_term = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

                for i, feature in enumerate(a_new_sto):
                    mean = feature[0]
                    log_std = feature[1]
                    std = torch.exp(log_std)
                    var = std**2
                    
                    sum = sum + (a_new_pretanh[i] - mean)**2 / var # (x - mean)^2 / var
                    sum = sum + 2 * log_std + torch.log(torch.tensor(2 * torch.pi, requires_grad=True)) # log(2 * pi * var) = 2 * log(std) + log(2 * pi)
                    corrective_term = corrective_term - torch.log(1 - a_new[i]**2 + 1e-6) # -log(1 - tanh^2(a_new_pretanh)) (with epsilon to avoid division by zero)

                log_prob = -0.5 * sum + corrective_term # transformation-corrected log probability

                # ------------------------------------- CLARIFICATION -------------------------------------
                # Each loss is 0.5 * (prediction - target)^2 = 0.5 * MSE(prediction, target)
                # It is not the same the target VALUE of v (in a certain step) and the target NETWORK of v
                # -----------------------------------------------------------------------------------------

                # Reshape the augmented state to input to the target V-network as a single 1D tensor
                aug_state_next_1D = torch.cat([aug_state_next[0], aug_state_next[1]], dim=2).view(-1)

                # Target value for each loss
                with torch.no_grad():
                    target_v = qmin - self.temperature * log_prob
                    target_q = r + self.discount * vtg(aug_state_next_1D)

                # Set the gradients to zero
                optimizer_v.zero_grad()
                optimizer_q1.zero_grad()
                optimizer_q2.zero_grad()
                optimizer_pi.zero_grad()

                # Compute the losses
                J_v = 0.5 * F.mse_loss(v(aug_state_1D), target_v)
                J_q1 = 0.5 * F.mse_loss(q1(aug_state_1D, a), target_q)
                J_q2 = 0.5 * F.mse_loss(q2(aug_state_1D, a), target_q)
                J_pi = self.temperature * log_prob - qmin

                # Backpropagate
                J_v.backward()
                J_q1.backward(retain_graph=True)
                J_q2.backward(retain_graph=True)
                J_pi.backward(retain_graph=True)

                # Optimize parameters
                optimizer_v.step()
                optimizer_q1.step()
                optimizer_q2.step()
                optimizer_pi.step()

                # Soft update the target V-network
                with torch.no_grad():
                    for v_params, vtg_params in zip(v.parameters(), vtg.parameters()):
                        vtg_params.data.mul_(1 - self.tau)
                        vtg_params.data.add_(self.tau * v_params.data)

                if not g == 0:
                    sys.stdout.write("\033[F")
                print(f"Gradient step {g+1}/{self.gradient_steps} done!")

            print("✔ Iteration done!")