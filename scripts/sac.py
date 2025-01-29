import torch
import torchviz
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchrl.data import ReplayBuffer, ListStorage

import sys
import os
import warnings
from time import perf_counter
import pandas as pd
import matplotlib.pyplot as plt

from scripts.client import Client
from scripts.model import *
from scripts.utils import *

RT = 6371.0 # Earth radius in km

class Actor(nn.Module):
    """
    Class to represent an Actor (policy, model) in the context of the SAC algorithm. Children class of nn.Module.
    """
    def __init__(self, model: TransformerModelEOS, lr: float=1e-3):
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
    def __init__(self, conf: DataFromJSON, client: Client, save_path: str):
        self.__role_type = "Soft Actor-Critic"
        self.__conf = conf
        self.client = client
        self.save_path = save_path
        self.set_properties(conf)
        self.losses = {"v": [], "q1": [], "q2": [], "pi": []}
        self.tensor_manager = TensorManager()

    def __str__(self) -> str:
        return f"{self.__role_type} object with configuration: {self.__conf}"

    def set_properties(self, conf: DataFromJSON):
        """
        Set the properties of the SAC object.
        """
        for key, value in conf.__dict__.items():
            if not key.startswith("__"):
                setattr(self, key, value)

    def start(self):
        """
        Start the SAC algorithm.
        """
        # Create the agent and the critics
        actor, q1, q2, v, vtg = self.create_entities()

        # Warm up the agent
        list_states, list_actions = self.warm_up(actor)

        # Train the agent
        actor, q1, q2, v, vtg = self.train(actor, q1, q2, v, vtg, list_states, list_actions)

        # Save the model
        self.save_model(actor, q1, q2, v, vtg)

        # Plot the losses
        self.plot_losses(self.losses)

    def create_entities(self) -> tuple[Actor, QNetwork, QNetwork, VNetwork, VNetwork]:
        """
        Create the entities for the SAC algorithm.
        """
        # Add the configuration fiel properties of the architecture chosen
        for i in range(len(self.architectures_available)):
            if self.architectures_available[i]["name"] == self.architecture_used:
                architecture_conf = DataFromJSON(self.architectures_available[i], "architecture_conf")
                break

        self.set_properties(architecture_conf)

        # Select the exact configuration for the model
        if self.architecture_used == "Transformer":
            actor, q1, q2, v, vtg = self.create_transformer_entities()
        elif self.architecture_used == "MLP":
            actor, q1, q2, v, vtg = self.create_mlp_entities()

        return actor, q1, q2, v, vtg

    def create_transformer_entities(self) -> tuple[Actor, QNetwork, QNetwork, VNetwork, VNetwork]:
        """
        Create the entities for the SAC algorithm with the Transformer architecture.
        """
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
        model = TransformerModelEOS(
            state_embedder=states_embedder,
            action_embedder=actions_embedder,
            pos_encoder=pos_encoder,
            transformer=transformer,
            projector=stochastic_projector
        )

        # Create the actor
        actor = Actor(model, lr=self.lambda_pi)

        # Create the NNs for the Q-networks
        q1 = QNetwork((self.state_dim + self.action_dim) * self.max_len, self.action_dim, 1, lr=self.lambda_q)
        q2 = QNetwork((self.state_dim + self.action_dim) * self.max_len, self.action_dim, 1, lr=self.lambda_q)

        # Create the NNs for the V-networks
        v = VNetwork((self.state_dim + self.action_dim) * self.max_len, 1, lr=self.lambda_v)
        vtg = VNetwork((self.state_dim + self.action_dim) * self.max_len, 1, lr=self.lambda_v)

        # Load the previous models if they exist
        if os.path.exists(self.save_path) and self.load_model and os.path.exists(f"{self.save_path}/model.pth"):
            print("Loading previous models...")
            actor.model.load_state_dict(torch.load(f"{self.save_path}/model.pth", weights_only=True))
            q1.load_state_dict(torch.load(f"{self.save_path}/q1.pth", weights_only=True))
            q2.load_state_dict(torch.load(f"{self.save_path}/q2.pth", weights_only=True))
            v.load_state_dict(torch.load(f"{self.save_path}/v.pth", weights_only=True))
            vtg.load_state_dict(torch.load(f"{self.save_path}/vtg.pth", weights_only=True))

        if os.path.exists(self.save_path) and self.load_buffer and os.path.exists(f"{self.save_path}/buffer.pth"):
            print("Loading previous replay buffer...")
            with warnings.catch_warnings():
                # Ignore the FutureWarning about loading with pickle
                warnings.simplefilter("ignore", category=FutureWarning)
                # storage: ListStorage = torch.load(f"{self.save_path}/buffer.pth") # This is the old way for Windows
                plain_list = torch.load(f"{self.save_path}/buffer.pth")
                storage = ListStorage(max_size=self.replay_buffer_size)
                storage._storage = plain_list
            self.replay_buffer = ReplayBuffer(storage=storage)
        else:
            print("Creating new replay buffer...")
            storage = ListStorage(max_size=self.replay_buffer_size)
            self.replay_buffer = ReplayBuffer(storage=storage)

        return actor, q1, q2, v, vtg
    
    def create_mlp_entities(self):
        """
        Create the entities for the SAC algorithm with the MLP architecture.
        """
        # Create the MLP model
        model = MLPModelEOS(
            state_dim=self.state_dim * self.max_len,
            action_dim=self.action_dim,
            n_hidden=self.hidden_layers,
            dropout=self.dropout
        )

        # Create the actor
        actor = Actor(model, lr=self.lambda_pi)

        # Create the NNs for the Q-networks
        q1 = QNetwork((self.state_dim + self.action_dim) * self.max_len, self.action_dim, 1, lr=self.lambda_q)
        q2 = QNetwork((self.state_dim + self.action_dim) * self.max_len, self.action_dim, 1, lr=self.lambda_q)

        # Create the NNs for the V-networks
        v = VNetwork(self.state_dim * self.max_len, 1, lr=self.lambda_v)
        vtg = VNetwork(self.state_dim * self.max_len, 1, lr=self.lambda_v)

        # Load the previous models if they exist
        if os.path.exists(self.save_path) and self.load_model and os.path.exists(f"{self.save_path}/model.pth"):
            print("Loading previous models...")
            actor.model.load_state_dict(torch.load(f"{self.save_path}/model.pth", weights_only=True))
            q1.load_state_dict(torch.load(f"{self.save_path}/q1.pth", weights_only=True))
            q2.load_state_dict(torch.load(f"{self.save_path}/q2.pth", weights_only=True))
            v.load_state_dict(torch.load(f"{self.save_path}/v.pth", weights_only=True))
            vtg.load_state_dict(torch.load(f"{self.save_path}/vtg.pth", weights_only=True))

        if os.path.exists(self.save_path) and self.load_buffer and os.path.exists(f"{self.save_path}/buffer.pth"):
            print("Loading previous replay buffer...")
            with warnings.catch_warnings():
                # Ignore the FutureWarning about loading with pickle
                warnings.simplefilter("ignore", category=FutureWarning)
                # storage: ListStorage = torch.load(f"{self.save_path}/buffer.pth") # This is the old way for Windows
                plain_list = torch.load(f"{self.save_path}/buffer.pth")
                storage = ListStorage(max_size=self.replay_buffer_size)
                storage._storage = plain_list
            self.replay_buffer = ReplayBuffer(storage=storage)
        else:
            print("Creating new replay buffer...")
            storage = ListStorage(max_size=self.replay_buffer_size)
            self.replay_buffer = ReplayBuffer(storage=storage)

        return actor, q1, q2, v, vtg

    def warm_up(self, actor: Actor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Warm up the SAC algorithm before starting the training loop.
        """
        torch.autograd.set_detect_anomaly(True)

        list_states: list[torch.Tensor] = []
        list_actions: list[torch.Tensor] = []

        # Loop over all agents
        for agt in self.agents:
            # Sending data to get the initial state
            sending_data = {
                "agent_id": agt,
                "action": {
                    "d_pitch": 0,
                    "d_roll": 0
                },
                "delta_time": 0
            }
            state, _, _ = self.client.get_next_state("get_next", sending_data)

            # Normalize the state given by the environment
            vec_state = self.normalize_state(state)

            # Input tensor of 1 batch and 1 sequence of state_dim dimensional states
            states = torch.FloatTensor([[vec_state]])

            # Input tensor of 1 batch and 1 sequence of action_dim dimensional actions (equal to 0)
            actions = torch.FloatTensor([[[0 for _ in range(self.action_dim)]]])

            list_states += [states]
            list_actions += [actions]

        # Loop flags
        done = False

        print("Starting warm up...")

        # Number of warm up steps required to fill the replay buffer
        warm_up_steps = self.replay_buffer_size - len(self.replay_buffer)
        warm_up_steps = warm_up_steps // len(self.agents) + warm_up_steps % len(self.agents)

        # Loop over all iterations
        for w in range(warm_up_steps):
            # Loop over all agents
            for idx, agt in enumerate(self.agents):
                states = list_states[idx]
                actions = list_actions[idx]

                with torch.no_grad():
                    # Adjust the maximum length of the states and actions
                    states = states[:, -self.max_len:, :]
                    actions = actions[:, -self.max_len:, :]

                    if self.debug:
                        before = perf_counter()

                    # Create the augmented state
                    aug_state = [states.clone(), actions.clone()]

                    # Get the stochastic actions
                    stochastic_actions = actor(states, actions)

                    # Select the last stochastic action
                    a_sto = stochastic_actions[-1, -1, :]

                    # Sample and convert the action
                    _, a = actor.model.reparametrization_trick(a_sto)

                    if self.debug:
                        print(f"Time taken to get the action: {perf_counter() - before:.4f} seconds")

                    if self.debug:
                        before = perf_counter()

                    # --------------- Environment's job to provide info ---------------
                    sending_data = {
                        "agent_id": agt,
                        "action": {
                            "d_pitch": a[0].item() * 90,
                            "d_roll": a[1].item() * 180
                        },
                        "delta_time": self.time_increment
                    }
                    
                    state, reward, done = self.client.get_next_state("get_next", sending_data)

                    if self.debug:
                        print(f"Time taken to get the next state: {perf_counter() - before:.4f} seconds")

                    # Break if time is up
                    if done:
                        print("Time is up!")
                        break

                    # Normalize the state
                    vec_state = self.normalize_state(state)

                    # Get the reward
                    r = torch.FloatTensor([reward * self.reward_scale])

                    # Get the next state
                    s_next = torch.FloatTensor(vec_state)
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
                    self.replay_buffer.add((aug_state, a, r, aug_state_next))

                    # Replace the states and actions lists
                    list_states[idx] = states
                    list_actions[idx] = actions

            # Break if time is up
            if done:
                print("Time is up!")
                break

            if not w == 0 and not self.debug:
                sys.stdout.write("\033[F")
            print(f"Warm up step {w+1}/{warm_up_steps} done!")

        print("✔ Warm up done!")
        
        return list_states, list_actions

    def train(self, actor: Actor, q1: QNetwork, q2: QNetwork, v: VNetwork, vtg: VNetwork, list_states: list[torch.Tensor], list_actions: list[torch.Tensor]):
        """
        Begin the training of the SAC algorithm.
        """
        torch.autograd.set_detect_anomaly(True)

        # Optimizers
        optimizer_v = optim.Adam(v.parameters(), lr=v.lr)
        optimizer_q1 = optim.Adam(q1.parameters(), lr=q1.lr)
        optimizer_q2 = optim.Adam(q2.parameters(), lr=q2.lr)
        optimizer_pi = optim.Adam(actor.model.parameters(), lr=actor.lr)

        # Loop flags
        done = False
        iteration = 1

        print("Starting training...")

        # Loop over all iterations
        while not done:
            print(f"\nStarting iteration {iteration}...")
            iteration += 1

            # Loop over all environment steps
            for e in range(self.environment_steps):
                # Loop over all agents
                for idx, agt in enumerate(self.agents):
                    states = list_states[idx]
                    actions = list_actions[idx]

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
                        _, a = actor.model.reparametrization_trick(a_sto)

                        # --------------- Environment's job to provide info ---------------
                        sending_data = {
                            "agent_id": agt,
                            "action": {
                                "d_pitch": a[0].item() * 90,
                                "d_roll": a[1].item() * 180
                            },
                            "delta_time": self.time_increment
                        }
                        
                        state, reward, done = self.client.get_next_state("get_next", sending_data)

                        # Break if time is up
                        if done:
                            print("Time is up!")
                            break

                        # Normalize the state
                        vec_state = self.normalize_state(state)

                        # Get the reward
                        r = torch.FloatTensor([reward * self.reward_scale])

                        # Get the next state
                        s_next = torch.FloatTensor(vec_state)
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
                        self.replay_buffer.add((aug_state, a, r, aug_state_next))

                        # Replace the states and actions lists
                        list_states[idx] = states
                        list_actions[idx] = actions

                # Break if time is up
                if done:
                    break

                if not e == 0:
                    sys.stdout.write("\033[F")
                print(f"Environment step {e+1}/{self.environment_steps} done!")
            
            # Break if time is up
            if done:
                break

            # Loop over all gradient steps
            for g in range(self.gradient_steps):
                with torch.no_grad():
                    aug_state, a, r, aug_state_next = self.tensor_manager.full_squeeze(*self.replay_buffer.sample(1))

                # Batchify the tensors neccessary for the transformer
                aug_state, aug_state_next = self.tensor_manager.batchify(aug_state, aug_state_next)

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
                J_v: torch.Tensor = 0.5 * F.mse_loss(v(aug_state_1D), target_v)
                J_q1: torch.Tensor = 0.5 * F.mse_loss(q1(aug_state_1D, a), target_q)
                J_q2: torch.Tensor = 0.5 * F.mse_loss(q2(aug_state_1D, a), target_q)
                J_pi: torch.Tensor = self.temperature * log_prob - qmin

                if self.debug:
                    print(f"Reward: {r.item():.4f}; Losses: {J_v.item():.4f}, {J_q1.item():.4f}, {J_q2.item():.4f}, {J_pi.item():.4f}; Log probability: {log_prob.item():.4f}; Qmin: {qmin.item():.4f}")

                # Store the losses
                self.losses["v"].append(J_v.item())
                self.losses["q1"].append(J_q1.item())
                self.losses["q2"].append(J_q2.item())
                self.losses["pi"].append(J_pi.item())

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

                if not g == 0 and not self.debug:
                    sys.stdout.write("\033[F")
                print(f"Gradient step {g+1}/{self.gradient_steps} done!")

            print("✔ Iteration done!")

        return actor, q1, q2, v, vtg
    
    def normalize_state(self, state: dict) -> list:
        """
        Normalize the action dictionary to a list.
        """
        # Conversion dictionary: each has two elements, the first is the gain and the second is the offset
        conversion_dict = {
            "a": (1/RT, -1), "e": (1, 0), "i": (1/180, 0), "raan": (1/360, 0), "aop": (1/360, 0), "ta": (1/360, 0), # orbital elements
            "az": (1/360, 0), "el": (1/180, 0.5), # azimuth and elevation
            "pitch": (1/180, 0.5), "roll": (1/360, 0.5), # attitude
            "detic_lat": (1/180, 0.5), "detic_lon": (1/360, 0), "detic_alt": (1/RT, 0), # nadir position
            "lat": (1/180, 0.5), "lon": (1/360, 0), "priority": (1/10, 0) # targets clues
        }

        vec_state = []
        for key, value in state.items():
            if key.startswith("lat_") or key.startswith("lon_") or key.startswith("priority_"):
                key = key.split("_")[0]
            vec_state.append(value * conversion_dict[key][0] + conversion_dict[key][1])

        return vec_state
    
    def plot_losses(self, losses: dict):
        """
        Plot the losses.
        """
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        smoothed_q1 = pd.DataFrame(losses["q1"]).rolling(window=int(len(losses["q1"])/10)).mean()
        smoothed_q2 = pd.DataFrame(losses["q2"]).rolling(window=int(len(losses["q2"])/10)).mean()
        smoothed_v = pd.DataFrame(losses["v"]).rolling(window=int(len(losses["v"])/10)).mean()
        smoothed_pi = pd.DataFrame(losses["pi"]).rolling(window=int(len(losses["pi"])/10)).mean()

        ax[0, 0].plot(smoothed_v)
        ax[0, 0].set_title("V-network loss")

        ax[0, 1].plot(smoothed_q1)
        ax[0, 1].set_title("Q1-network loss")

        ax[1, 0].plot(smoothed_q2)
        ax[1, 0].set_title("Q2-network loss")

        ax[1, 1].plot(smoothed_pi)
        ax[1, 1].set_title("Policy loss")

        plt.savefig(f"{self.save_path}/losses.png", dpi=500)
    
    def save_model(self, actor: Actor, q1: QNetwork, q2: QNetwork, v: VNetwork, vtg: VNetwork):
        """
        Save the model to the specified path.
        """
        torch.save(actor.model.state_dict(), f"{self.save_path}/model.pth")
        torch.save(q1.state_dict(), f"{self.save_path}/q1.pth")
        torch.save(q2.state_dict(), f"{self.save_path}/q2.pth")
        torch.save(v.state_dict(), f"{self.save_path}/v.pth")
        torch.save(vtg.state_dict(), f"{self.save_path}/vtg.pth")
        torch.save(list(self.replay_buffer.storage), f"{self.save_path}/buffer.pth")