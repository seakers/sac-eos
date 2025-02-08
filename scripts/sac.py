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
    def __init__(self, in_dim: int, out_dim: int, n_hidden: tuple[int], lr: float=1e-3):
        super(Critic, self).__init__()
        self.role_type = "Critic"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hidden = n_hidden
        self.lr = lr

        layers = []
        layers.append(nn.Linear(in_dim, n_hidden[0]))
        layers.append(nn.ReLU())

        for i in range(len(n_hidden) - 1):
            layers.append(nn.Linear(n_hidden[i], n_hidden[i + 1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.LayerNorm(n_hidden[-1]))
        layers.append(nn.Linear(n_hidden[-1], out_dim))
        self.mlp = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, x):
        return self.mlp(x)

class QNetwork(Critic):
    """
    Class to represent a Q-network. Children class of Critic.
    """
    def __init__(self, state_dim: int, action_dim: int, max_len: int, out_dim: int, n_hidden: tuple[int], lr: float=1e-3, aug_state_contains_actions: bool=False):
        # Adjust the size of the augmented state based on the architecture
        if aug_state_contains_actions:
            aug_state_size = (state_dim + action_dim) * max_len
        else:
            aug_state_size = state_dim * max_len
        
        # Create the class with the parent class initializer
        super(QNetwork, self).__init__(in_dim=(aug_state_size + action_dim), out_dim=out_dim, n_hidden=n_hidden, lr=lr)
        self.critic_type = "Q-network"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_len = max_len
        self.aug_state_contains_actions = aug_state_contains_actions
        self.gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, states, actions, new_action):
        if states.shape[-2] != actions.shape[-2]:
            raise ValueError("The states and actions sequences must have the same length!")

        # Fill the state and actions tensors so that there are max_len elements
        if states.shape[-2] < self.max_len:
            states = torch.cat([states, torch.zeros(states.shape[0], self.max_len - states.shape[-2], states.shape[-1], device=self.gpu_device)], dim=-2)
            actions = torch.cat([actions, torch.zeros(actions.shape[0], self.max_len - actions.shape[-2], actions.shape[-1], device=self.gpu_device)], dim=-2)

        if self.aug_state_contains_actions:
            aug_state_1D = torch.cat([states, actions], dim=2).view(-1)
        else:
            aug_state_1D = states.view(-1)

        x = torch.cat([aug_state_1D, new_action])
        x = super(QNetwork, self).forward(x)
        return x
    
class VNetwork(Critic):

    """
    Class to represent a V-network. Children class of Critic.
    """
    def __init__(self, state_dim: int, action_dim: int, max_len: int, out_dim: int, n_hidden: tuple[int], lr: float=1e-3, aug_state_contains_actions: bool=False):
        # Adjust the size of the augmented state based on the architecture
        if aug_state_contains_actions:
            aug_state_size = (state_dim + action_dim) * max_len
        else:
            aug_state_size = state_dim * max_len

        super(VNetwork, self).__init__(in_dim=aug_state_size, out_dim=out_dim, n_hidden=n_hidden, lr=lr)
        self.critic_type = "V-network"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_len = max_len
        self.aug_state_contains_actions = aug_state_contains_actions
        self.gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, states, actions):
        if states.shape[-2] != actions.shape[-2]:
            raise ValueError("The states and actions sequences must have the same length!")

        # Fill the state and actions tensors so that there are max_len elements
        if states.shape[-2] < self.max_len:
            states = torch.cat([states, torch.zeros(states.shape[0], self.max_len - states.shape[-2], states.shape[-1], device=self.gpu_device)], dim=-2)
            actions = torch.cat([actions, torch.zeros(actions.shape[0], self.max_len - actions.shape[-2], actions.shape[-1], device=self.gpu_device)], dim=-2)

        if self.aug_state_contains_actions:
            aug_state_1D = torch.cat([states, actions], dim=2).view(-1)
        else:
            aug_state_1D = states.view(-1)

        x = aug_state_1D
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
        self.gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_properties(conf)
        self.losses = {"v": [], "q1": [], "q2": [], "pi": []}
        self.tensor_manager = TensorManager()
        self.a_conversions = torch.tensor(self.a_conversions) # action conversions from transformer post-tanh to real action

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

        # Move items to gpu
        actor = actor.to(self.gpu_device)
        q1 = q1.to(self.gpu_device)
        q2 = q2.to(self.gpu_device)
        v = v.to(self.gpu_device)
        vtg = vtg.to(self.gpu_device)

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

        # Set scaling factor
        self.scaling_factor = actor.model.scaling_factor

        return actor, q1, q2, v, vtg

    def create_transformer_entities(self) -> tuple[Actor, QNetwork, QNetwork, VNetwork, VNetwork]:
        """
        Create the entities for the SAC algorithm with the Transformer architecture.
        """
        # Create the embedder object for states
        states_embedder = FloatEmbedder(
            input_dim=self.state_dim,
            embed_dim=self.d_model,
            dropout=self.embed_dropout
        )
        
        # Create the embedder object for actions
        actions_embedder = FloatEmbedder(
            input_dim=self.action_dim,
            embed_dim=self.d_model,
            dropout=self.embed_dropout
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
            batch_first=self.batch_first,
            kaiming_init=self.kaiming_init
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
            projector=stochastic_projector,
            a_conversions=self.a_conversions
        )

        # Create the actor
        actor = Actor(model, lr=self.lambda_pi)

        # Create the NNs for the Q-networks
        q1, q2, v, vtg = self.create_nn_critics(self.aug_state_contains_actions)

        # Load the previous models if they exist
        actor, q1, q2, v, vtg = self.load_previous_models(actor, q1, q2, v, vtg)

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
            dropout=self.dropout,
            a_conversions=self.a_conversions
        )

        # Create the actor
        actor = Actor(model, lr=self.lambda_pi)

        # Create the NNs for the Q-networks
        q1, q2, v, vtg = self.create_nn_critics(self.aug_state_contains_actions)

        # Load the previous models if they exist
        actor, q1, q2, v, vtg = self.load_previous_models(actor, q1, q2, v, vtg)

        return actor, q1, q2, v, vtg
    
    def create_nn_critics(self, aug_state_contains_actions: bool=True) -> tuple[Actor, QNetwork, QNetwork, VNetwork, VNetwork]:
        """
        Create the neural networks for the Q-networks and the V-networks.
        """
        # Create the NNs for the Q-networks
        q1 = QNetwork(self.state_dim, self.action_dim, self.max_len, 1, n_hidden=self.critics_hidden_layers, lr=self.lambda_q, aug_state_contains_actions=aug_state_contains_actions)
        q2 = QNetwork(self.state_dim, self.action_dim, self.max_len, 1, n_hidden=self.critics_hidden_layers, lr=self.lambda_q, aug_state_contains_actions=aug_state_contains_actions)

        # Create the NNs for the V-networks
        v = VNetwork(self.state_dim, self.action_dim, self.max_len, 1, n_hidden=self.critics_hidden_layers, lr=self.lambda_v, aug_state_contains_actions=aug_state_contains_actions)
        vtg = VNetwork(self.state_dim, self.action_dim, self.max_len, 1, n_hidden=self.critics_hidden_layers, lr=self.lambda_v, aug_state_contains_actions=aug_state_contains_actions)

        # Set the vtg network to the same weights as the v network
        vtg.load_state_dict(v.state_dict())

        return q1, q2, v, vtg
    
    def load_previous_models(self, actor: Actor, q1: nn.Module, q2: nn.Module, v: nn.Module, vtg: nn.Module):
        """
        Load the previous models if they exist.
        """
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
        warm_up_steps = self.minimum_samples - len(self.replay_buffer)
        warm_up_steps = warm_up_steps if warm_up_steps > 0 else 0
        warm_up_steps = warm_up_steps // len(self.agents) + warm_up_steps % len(self.agents)

        # Loop over all iterations
        for w in range(warm_up_steps):
            # Loop over all agents
            for idx, agt in enumerate(self.agents):
                states = list_states[idx]
                actions = list_actions[idx]

                with torch.no_grad():
                    # Adjust the maximum length of the states and actions
                    states = states[:, -self.max_len:, :].to(self.gpu_device)
                    actions = actions[:, -self.max_len:, :].to(self.gpu_device)

                    if self.debug:
                        before = perf_counter()

                    # Create the augmented state
                    aug_state = [states.clone(), actions.clone()]

                    # Get the stochastic actions
                    stochastic_actions = actor(states, actions)

                    # Select the last stochastic action
                    a_sto = stochastic_actions[-1, -1, :]

                    # Sample and convert the action
                    _, a, a_norm = actor.model.reparametrization_trick(a_sto)

                    if self.debug:
                        print(f"Time taken to get the action: {perf_counter() - before:.4f} seconds")

                    if self.debug:
                        before = perf_counter()

                    # --------------- Environment's job to provide info ---------------
                    sending_data = {
                        "agent_id": agt,
                        "action": {
                            "d_pitch": a[0].item(),
                            "d_roll": a[1].item()
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
                    states = torch.cat([states, s_next.unsqueeze(0).unsqueeze(0).to(self.gpu_device)], dim=1)

                    # Add it to the actions
                    actions = torch.cat([actions, a_norm.unsqueeze(0).unsqueeze(0)], dim=1)

                    # Adjust the maximum length of the states and actions
                    states = states[:, -self.max_len:, :]
                    actions = actions[:, -self.max_len:, :]

                    # Augmented state for the next step
                    aug_state_next = [states, actions]

                    # Store in the buffer
                    self.replay_buffer.add((aug_state, a_norm, r, aug_state_next))

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
                        states = states[:, -self.max_len:, :].to(self.gpu_device)
                        actions = actions[:, -self.max_len:, :].to(self.gpu_device)

                        # Create the augmented state
                        aug_state = [states.clone(), actions.clone()]

                        # Get the stochastic actions
                        stochastic_actions = actor(states, actions)

                        # Select the last stochastic action
                        a_sto = stochastic_actions[-1, -1, :]

                        # Sample and convert the action
                        _, a, a_norm = actor.model.reparametrization_trick(a_sto)

                        # --------------- Environment's job to provide info ---------------
                        sending_data = {
                            "agent_id": agt,
                            "action": {
                                "d_pitch": a[0].item(),
                                "d_roll": a[1].item()
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
                        states = torch.cat([states, s_next.unsqueeze(0).unsqueeze(0).to(self.gpu_device)], dim=1)

                        # Add it to the actions
                        actions = torch.cat([actions, a_norm.unsqueeze(0).unsqueeze(0)], dim=1)

                        # Adjust the maximum length of the states and actions
                        states = states[:, -self.max_len:, :]
                        actions = actions[:, -self.max_len:, :]

                        # Augmented state for the next step
                        aug_state_next = [states, actions]

                        # Store in the buffer
                        self.replay_buffer.add((aug_state, a_norm, r, aug_state_next))

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
                    aug_state, a_norm, r, aug_state_next = self.tensor_manager.full_squeeze(*self.replay_buffer.sample(1))

                # Batchify the tensors neccessary for the transformer
                aug_state, aug_state_next = self.tensor_manager.batchify(aug_state, aug_state_next)

                # Get the stochastic actions again
                new_stochastic_actions = actor(aug_state[0].to(self.gpu_device), aug_state[1].to(self.gpu_device))

                # Select the last stochastic action
                a_new_sto = new_stochastic_actions[-1, -1, :]

                # Sample and convert the action
                a_new_preconv, _, a_new_norm = actor.model.reparametrization_trick(a_new_sto)

                # Find the minimum of the Q-networks for the replay buffer sample and the new action
                q1_replay = q1(aug_state[0], aug_state[1], a_norm)
                q2_replay = q2(aug_state[0], aug_state[1], a_norm)
                qmin_new = torch.min(q1(aug_state[0], aug_state[1], a_new_norm), q2(aug_state[0], aug_state[1], a_new_norm))

                k = 1 / self.scaling_factor
                corrective_terms = k / torch.cosh(a_new_preconv / self.scaling_factor)**2 # 1 - tanh^2 = sech^2 = 1 / cosh^2
                normal_dist = torch.distributions.Normal(a_new_sto[:, 0], torch.exp(a_new_sto[:, 1]))
                log_prob = normal_dist.log_prob(a_new_preconv).sum() - torch.log(torch.clamp(corrective_terms, min=1e-5)).sum()

                if self.debug:
                    print("LogProbDen:", normal_dist.log_prob(a_new_preconv).sum(), "Corrective terms:", corrective_terms, "-log(corrective):", - torch.log(torch.clamp(corrective_terms, min=1e-5)).sum())

                # ------------------------------------- CLARIFICATION -------------------------------------
                # Each loss is 0.5 * (prediction - target)^2 = 0.5 * MSE(prediction, target)
                # It is not the same the target VALUE of v (in a certain step) and the target NETWORK of v
                # -----------------------------------------------------------------------------------------

                # Target value for each loss
                with torch.no_grad():
                    target_v = qmin_new - self.temperature * log_prob
                    target_q = r + self.discount * vtg(aug_state_next[0], aug_state_next[1])

                # Set the gradients to zero
                optimizer_v.zero_grad()
                optimizer_q1.zero_grad()
                optimizer_q2.zero_grad()
                optimizer_pi.zero_grad()

                # Compute the losses
                J_v: torch.Tensor = 0.5 * F.mse_loss(v(aug_state[0], aug_state[1]), target_v)
                J_q1: torch.Tensor = 0.5 * F.mse_loss(q1_replay, target_q)
                J_q2: torch.Tensor = 0.5 * F.mse_loss(q2_replay, target_q)
                J_pi: torch.Tensor = self.temperature * log_prob - qmin_new

                if self.debug:
                    print("V ----> Loss:", f"{J_v.item():.3f}", "Forward:", f"{v(aug_state[0], aug_state[1]).item():.3f}", "Target:", f"{target_v.item():.3f}", "Qmin:", f"{qmin_new.item():.3f}")
                    print("Q1 ---> Loss:", f"{J_q1.item():.3f}", "Forward:", f"{q1_replay.item():.3f}", "Target:", f"{target_q.item():.3f}")
                    print("Q2 ---> Loss:", f"{J_q2.item():.3f}", "Forward:", f"{q2_replay.item():.3f}", "Target:", f"{target_q.item():.3f}")
                    print("Pi ---> Loss:", f"{J_pi.item():.3f}", "Qmin:", f"{qmin_new.item():.3f}", "Alpha:", f"{self.temperature:.3f}", "LogProbDen:", f"{log_prob.item():.3f}")
                    print("Vtg --> Forward:", f"{vtg(aug_state_next[0], aug_state_next[1]).item():.3f}", "Reward:", f"{r.item():.3f}")

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
        Normalize the state dictionary to a list.
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