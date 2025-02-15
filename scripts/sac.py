import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.data import ReplayBuffer, ListStorage

from types import SimpleNamespace

import sys
import os
import warnings
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
    def __init__(self, model: nn.Module, architecture: str, lr: float=1e-3):
        super(Actor, self).__init__()
        self.__role_type = "Actor"
        self.model = model
        self.architecture = architecture
        self.lr = lr

    def forward(self, states, actions):
        if self.architecture == "Transformer":
            return self.model(states, actions)
        elif self.architecture == "TransformerEncoder":
            return self.model(states)
        elif self.architecture == "MLP":
            return self.model(states)
        
class QNetwork(nn.Module):
    """
    Class to represent a Q-network. Children class of nn.Module.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_len: int,
        n_atoms: int,
        truncated_atoms: int,
        model: nn.Module,
        architecture: str,
        lr: float,
        obs_has_actions: bool
    ):
        super(QNetwork, self).__init__()
        self.__role_type = "Q-network"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_len = max_len
        self.n_atoms = n_atoms
        self.truncated_atoms = truncated_atoms
        self.model = model
        self.architecture = architecture
        self.lr = lr
        self.obs_has_actions = obs_has_actions

        self.mlp = nn.Sequential(
            nn.Linear(model.out_dim + action_dim, n_atoms),
            nn.ReLU()
        )

    def forward(self, states, actions, new_action):
        if states.shape[-2] != actions.shape[-2]:
            raise ValueError("The states and actions sequences must have the same length!")

        # Fill the state and actions tensors so that there are max_len elements
        if states.shape[-2] < self.max_len:
            states = torch.cat([states, torch.zeros(states.shape[0], self.max_len - states.shape[-2], states.shape[-1], device=self.gpu_device)], dim=-2)
            actions = torch.cat([actions, torch.zeros(actions.shape[0], self.max_len - actions.shape[-2], actions.shape[-1], device=self.gpu_device)], dim=-2)

        if self.architecture == "Transformer":
            x = self.model(states, actions)[:, -1, :]
            x = self.mlp(torch.cat([x, new_action], dim=-1))
        elif self.architecture == "TransformerEncoder":
            x = self.model(states)[:, -1, :]
            x = self.mlp(torch.cat([x, new_action], dim=-1))
        elif self.architecture == "MLP":
            if self.obs_has_actions:
                x = torch.cat([states, actions], dim=-1).view(states.shape[0], -1)
            else:
                x = states.view(states.shape[0], -1)
                x = torch.cat([x, new_action], dim=-1)
            x = self.model(x)
        
        # Perform the truncation
        x, _ = torch.topk(x, k=self.truncated_atoms, dim=-1, largest=False)
        x = x.mean(dim=-1)
        # Make sure the output is a 2D tensor
        return x.view(x.shape[0], -1)
        
class VNetwork(nn.Module):
    """
    Class to represent a V-network. Children class of nn.Module.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_len: int,
        n_atoms: int,
        truncated_atoms: int,
        model: nn.Module,
        architecture: str,
        lr: float,
        obs_has_actions: bool
    ):
        super(VNetwork, self).__init__()
        self.__role_type = "V-network"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_len = max_len
        self.n_atoms = n_atoms
        self.truncated_atoms = truncated_atoms
        self.model = model
        self.architecture = architecture
        self.lr = lr
        self.obs_has_actions = obs_has_actions

    def forward(self, states, actions):
        if states.shape[-2] != actions.shape[-2]:
            raise ValueError("The states and actions sequences must have the same length!")

        # Fill the state and actions tensors so that there are max_len elements
        if states.shape[-2] < self.max_len:
            states = torch.cat([states, torch.zeros(states.shape[0], self.max_len - states.shape[-2], states.shape[-1], device=self.gpu_device)], dim=-2)
            actions = torch.cat([actions, torch.zeros(actions.shape[0], self.max_len - actions.shape[-2], actions.shape[-1], device=self.gpu_device)], dim=-2)

        if self.architecture == "Transformer":
            x = self.model(states, actions)[:, -1, :]
        elif self.architecture == "TransformerEncoder":
            x = self.model(states)[:, -1, :]
        elif self.architecture == "MLP":
            if self.obs_has_actions:
                x = torch.cat([states, actions], dim=-1).view(states.shape[0], -1)
            else:
                x = states.view(states.shape[0], -1)
            x = self.model(x)

        # Perform the truncation
        x, _ = torch.topk(x, k=self.truncated_atoms, dim=-1, largest=False)
        x = x.mean(dim=-1)
        # Make sure the output is a 2D tensor
        return x.view(x.shape[0], -1)

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
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.losses = {"v": [], "q1": [], "q2": [], "pi": []}
        self.tensor_manager = TensorManager()
        self.a_conversions = torch.tensor(self.a_conversions) # action conversions from transformer post-tanh to real action space

    def __str__(self) -> str:
        return f"{self.__role_type} object with configuration: {self.__conf}"

    def set_properties(self, conf: DataFromJSON, container: object=None):
        """
        Set the properties of the SAC object.
        """
        for key, value in conf.__dict__.items():
            if not key.startswith("__"):
                setattr(container if container is not None else self, key, value)

    def start(self):
        """
        Start the SAC algorithm.
        """
        # Create the actor
        actor = self.create_actor()

        # Create the replay buffer
        self.create_replay_buffer()

        if self.algo_version == "Original":
            # Create the critics
            q1, q2, v, vtg = self.create_the_critics(["q1", "q2", "v", "vtg"], self.obs_has_actions)

            # Warm up the agent
            list_states, list_actions = self.warm_up(actor)
            
            # Train with the original algorithm
            actor, q1, q2, v, vtg = self.train_original(actor, q1, q2, v, vtg, list_states, list_actions)

            # Save the model
            self.save_parameters({"model": actor.model, "q1": q1, "q2": q2, "v": v, "vtg": vtg})
        elif self.algo_version == "OpenAI":
            # Create the critics
            q1, q1tg, q2, q2tg = self.create_the_critics(["q1", "q1tg", "q2", "q2tg"], self.obs_has_actions)

            # Warm up the agent
            list_states, list_actions = self.warm_up(actor)

            # Train with the OpenAI algorithm
            actor, q1, q1tg, q2, q2tg = self.train_openai(actor, q1, q1tg, q2, q2tg, list_states, list_actions)

            # Save the model
            self.save_parameters({"model": actor.model, "q1": q1, "q1tg": q1tg, "q2": q2, "q2tg": q2tg})
        else:
            raise ValueError("The version of the SAC algorithm is not recognized. Please try 'Original' or 'openAI'.")
        
        # Save the replay buffer
        self.save_replay_buffer()

        # Plot the losses
        self.plot_losses(self.losses)

    def create_actor(self) -> Actor:
        """
        Create the entities for the SAC algorithm.
        """
        # Add the configuration file properties of the architecture chosen
        for i in range(len(self.architectures_available)):
            if self.architectures_available[i]["name"] == self.architecture_used:
                architecture_conf = DataFromJSON(self.architectures_available[i], "architecture_conf")
                break

        self._actor_conf = SimpleNamespace()

        # Set the properties of the architecture
        self.set_properties(architecture_conf, self._actor_conf)

        # Assign actor to the Soft Actor-Critic object for generic use
        self.max_len = self._actor_conf.max_len
        self.obs_has_actions = self._actor_conf.obs_has_actions

        # Generic properties
        self._actor_conf.stochastic = True

        # Select the exact configuration for the model
        if self.architecture_used == "Transformer":
            self._actor_conf.src_dim = self.state_dim
            self._actor_conf.tgt_dim = self.action_dim
            self._actor_conf.out_dim = self.action_dim
            model = self.generate_transformer_model(self._actor_conf)
        elif self.architecture_used == "TransformerEncoder":
            self._actor_conf.src_dim = self.state_dim
            self._actor_conf.out_dim = self.action_dim
            model = self.generate_transformer_encoder_model(self._actor_conf)
        elif self.architecture_used == "MLP":
            self._actor_conf.in_dim = self.state_dim * self.max_len
            self._actor_conf.out_dim = self.action_dim
            model = self.generate_mlp_model(self._actor_conf)
        else:
            raise ValueError("The architecture used is not recognized. Please try 'Transformer' or 'MLP'.")

        # Set scaling factor
        self.scaling_factor = model.scaling_factor

        actor = Actor(
            model=model,
            architecture=self.architecture_used,
            lr=self.lr_pi
        )

        # Load the model if it exists
        self.load_parameters({"model": actor.model})

        return actor.to(self.gpu_device)
    
    def create_the_critics(self, critics: tuple[str], obs_has_actions: bool=True) -> tuple[nn.Module, ...]:
        """
        Create the desired critics for the SAC algorithm.
        """
        _critics: tuple[nn.Module, ...] = []

        # Add the configuration file properties of the architectures chosen
        for i in range(len(self.architectures_available)):
            if self.architectures_available[i]["name"] == self.q_net_architecture:
                q_conf = DataFromJSON(self.architectures_available[i], "architecture_conf")
            if self.architectures_available[i]["name"] == self.v_net_architecture:
                v_conf = DataFromJSON(self.architectures_available[i], "architecture_conf")

        self._q_conf = SimpleNamespace()
        self._v_conf = SimpleNamespace()

        # Set the properties of the architectures
        self.set_properties(q_conf, self._q_conf)
        self.set_properties(v_conf, self._v_conf)

        # Set generic properties
        self._q_conf.lr = self.lr_q
        self._v_conf.lr = self.lr_v
        self._q_conf.stochastic = False
        self._v_conf.stochastic = False     

        for critic in critics:
            if critic.startswith("q"):
                if self.q_net_architecture == "Transformer":
                    self._q_conf.src_dim = self.state_dim
                    self._q_conf.tgt_dim = self.action_dim
                    self._q_conf.out_dim = self.q_net_mid_dim - self.action_dim
                    if not obs_has_actions:
                        raise PermissionError("Q-networks transformers require the actions in the observations.")
                    model = self.generate_transformer_model(self._q_conf)
                elif self.q_net_architecture == "TransformerEncoder":
                    self._q_conf.src_dim = self.state_dim
                    self._q_conf.out_dim = self.q_net_mid_dim - self.action_dim
                    model = self.generate_transformer_encoder_model(self._q_conf)
                elif self.q_net_architecture == "MLP":
                    self._q_conf.in_dim = self.state_dim * self.max_len + self.action_dim
                    self._q_conf.out_dim = self.critics_atoms
                    model = self.generate_mlp_model(self._q_conf)
                else:
                    raise ValueError("The architecture used is not recognized. Please try 'Transformer' or 'MLP'.")
                _critics.append(QNetwork(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    max_len=self.max_len,
                    n_atoms=self.critics_atoms,
                    truncated_atoms=self.truncated_atoms,
                    model=model,
                    architecture=self.q_net_architecture,
                    lr=self.lr_q,
                    obs_has_actions=obs_has_actions
                ))
            if critic.startswith("v"):
                if self.v_net_architecture == "Transformer":
                    self._v_conf.src_dim = self.state_dim
                    self._v_conf.tgt_dim = self.action_dim
                    self._v_conf.out_dim = self.critics_atoms
                    if not obs_has_actions:
                        raise PermissionError("V-networks transformers require the actions in the observations.")
                    model = self.generate_transformer_model(self._v_conf)
                elif self.v_net_architecture == "TransformerEncoder":
                    self._v_conf.src_dim = self.state_dim
                    self._v_conf.out_dim = self.critics_atoms
                    model = self.generate_transformer_encoder_model(self._v_conf)
                elif self.v_net_architecture == "MLP":
                    self._v_conf.in_dim = self.state_dim * self.max_len
                    self._v_conf.out_dim = self.critics_atoms
                    model = self.generate_mlp_model(self._v_conf)
                else:
                    raise ValueError("The architecture used is not recognized. Please try 'Transformer' or 'MLP'.")
                _critics.append(VNetwork(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    max_len=self.max_len,
                    n_atoms=self.critics_atoms,
                    truncated_atoms=self.truncated_atoms,
                    model=model,
                    architecture=self.v_net_architecture,
                    lr=self.lr_v,
                    obs_has_actions=obs_has_actions
                ))

        # Critics list to dict of indices
        critics_dict: dict = {critic: i for i, critic in enumerate(critics)}

        # Set the target networks to the same weights as their normal network
        if self.load_params:
            self.load_parameters({name: critic for name, critic in zip(critics, _critics)})
        else:
            for name, i in critics_dict.items():
                if name.endswith("tg"):
                    j = critics_dict[name.split("tg")[0]]
                    _critics[i].load_state_dict(_critics[j].state_dict())

        # Put models on the GPU
        for critic in _critics:
            critic.to(self.gpu_device)

        return _critics

    def generate_transformer_model(self, base: object=None) -> TransformerModelEOS:
        """
        Create the actor for the SAC algorithm with the Transformer architecture.
        """
        base = base if base is not None else self
        return TransformerModelEOS(
            src_dim=base.src_dim,
            tgt_dim=base.tgt_dim,
            out_dim=base.out_dim,
            d_model=base.d_model,
            nhead=base.nhead,
            max_len=base.max_len,
            num_encoder_layers=base.num_encoder_layers,
            num_decoder_layers=base.num_decoder_layers,
            dim_feedforward=base.dim_feedforward,
            embed_dropout=base.embed_dropout,
            pos_dropout=base.pos_dropout,
            transformer_dropout=base.transformer_dropout,
            position_encoding=base.position_encoding,
            activation=base.activation,
            stochastic=base.stochastic,
            batch_first=base.batch_first,
            kaiming_init=base.kaiming_init,
            a_conversions=self.a_conversions
        )
    
    def generate_transformer_encoder_model(self, base: object=None) -> TransformerEncoderModelEOS:
        """
        Create the actor for the SAC algorithm with the TransformerEncoder architecture.
        """
        base = base if base is not None else self
        return TransformerEncoderModelEOS(
            src_dim=base.src_dim,
            out_dim=base.out_dim,
            d_model=base.d_model,
            nhead=base.nhead,
            max_len=base.max_len,
            num_encoder_layers=base.num_encoder_layers,
            dim_feedforward=base.dim_feedforward,
            embed_dropout=base.embed_dropout,
            pos_dropout=base.pos_dropout,
            encoder_dropout=base.encoder_dropout,
            position_encoding=base.position_encoding,
            activation=base.activation,
            stochastic=base.stochastic,
            batch_first=base.batch_first,
            kaiming_init=base.kaiming_init,
            a_conversions=self.a_conversions
        )
    
    def generate_mlp_model(self, base: object=None) -> MLPModelEOS:
        """
        Create the actor for the SAC algorithm with the MLP architecture.
        """
        base = base if base is not None else self
        return MLPModelEOS(
            in_dim=base.in_dim,
            out_dim=base.out_dim,
            hidden_layers=base.hidden_layers,
            dropout=base.dropout,
            stochastic=base.stochastic,
            a_conversions=self.a_conversions
        )
    
    def create_replay_buffer(self) -> None:
        """
        Load the previous models if they exist.
        """
        # Load the previous models if they exist
        if self.load_buffer and os.path.exists(f"{self.save_path}/buffer.pth"):
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

                done, next_states, next_actions = self.do_1_experience(actor, states, actions, agt)

                # Break if time is up
                if done:
                    break

                # Replace the states and actions lists
                list_states[idx] = next_states
                list_actions[idx] = next_actions

            # Break if time is up
            if done:
                print("Time is up!")
                break

            if not w == 0 and not self.debug:
                sys.stdout.write("\033[F")
            print(f"Warm up step {w+1}/{warm_up_steps} done!")

        print("✔ Warm up done!")
        
        return list_states, list_actions
    
    def do_1_experience(self, actor: Actor, states: torch.Tensor, actions: torch.Tensor, agt: int):
        """
        Do an environment step for the SAC algorithm.
        """
        with torch.no_grad():
            # Adjust the maximum length of the states and actions
            states = states[:, -self.max_len:, :].to(self.gpu_device)
            actions = actions[:, -self.max_len:, :].to(self.gpu_device)

            # Get the stochastic actions
            stochastic_actions = actor(states, actions)

            # Select the last stochastic action
            a_sto = stochastic_actions[-1, -1, :, :] # get a 2-dim tensor with the last stochastic action

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
                return True, None, None

            # Normalize the state
            vec_state = self.normalize_state(state)

            # Get the reward
            r = torch.FloatTensor([reward * self.reward_scale])

            # Get the next state
            s_next = torch.FloatTensor(vec_state)
            # --------------- Environment's job to provide info ---------------

            # Add it to the states
            next_states = torch.cat([states, s_next.unsqueeze(0).unsqueeze(0).to(self.gpu_device)], dim=1)

            # Add it to the actions
            next_actions = torch.cat([actions, a_norm.unsqueeze(0).unsqueeze(0)], dim=1)

            # Adjust the maximum length of the states and actions
            next_states = next_states[:, -self.max_len:, :]
            next_actions = next_actions[:, -self.max_len:, :]

            # Store in the buffer only if the states and actions have the maximum length (or batch processing will collapse)
            if states.shape[-2] == self.max_len:
                self.replay_buffer.add((states.squeeze(0), actions.squeeze(0), a_norm, r, next_states.squeeze(0), next_actions.squeeze(0))) # batch sampling will unsqueeze the first dimension

            return False, next_states, next_actions

    def train_original(self, actor: Actor, q1: QNetwork, q2: QNetwork, v: VNetwork, vtg: VNetwork, list_states: list[torch.Tensor], list_actions: list[torch.Tensor]):
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

        print("Starting training with the original algorithm...")

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

                    # Do an environment step
                    done, next_states, next_actions = self.do_1_experience(actor, states, actions, agt)

                    # Break if time is up
                    if done:
                        break

                    # Replace the states and actions lists
                    list_states[idx] = next_states
                    list_actions[idx] = next_actions

                # Break if time is up
                if done:
                    break

                if not e == 0 and not self.debug:
                    sys.stdout.write("\033[F")
                print(f"Environment step {e+1}/{self.environment_steps} done!")
            
            # Break if time is up
            if done:
                break

            # Loop over all gradient steps
            for g in range(self.gradient_steps):
                with torch.no_grad():
                    states, actions, a_norm, r, next_states, next_actions = self.tensor_manager.full_squeeze(*self.replay_buffer.sample(self.batch_size))

                if self.debug:
                    print("Shapes of buffer sampling:", states.shape, actions.shape, a_norm.shape, r.shape, next_states.shape, next_actions.shape)

                # Batchify the tensors neccessary for the transformer
                states, actions, next_states, next_actions = self.tensor_manager.batchify(states, actions, next_states, next_actions) # batchify if necessary

                # Get the stochastic actions again
                new_stochastic_actions = actor(states.to(self.gpu_device), actions.to(self.gpu_device))

                # Select the last stochastic action
                a_new_sto = new_stochastic_actions[:, -1, :, :]

                # Sample and convert the action
                a_new_preconv, _, a_new_norm = actor.model.reparametrization_trick(a_new_sto)

                # Find the minimum of the Q-networks for the replay buffer sample and the new action
                v_replay: torch.Tensor = v(states, actions)
                vtg_replay_next: torch.Tensor = vtg(next_states, next_actions)
                q1_replay: torch.Tensor = q1(states, actions, a_norm)
                q2_replay: torch.Tensor = q2(states, actions, a_norm)
                qmin_new: torch.Tensor = torch.min(q1(states, actions, a_new_norm), q2(states, actions, a_new_norm))

                k = 1 / self.scaling_factor
                corrective_terms = k / torch.cosh(a_new_preconv / self.scaling_factor)**2 # 1 - tanh^2 = sech^2 = 1 / cosh^2
                normal_dist = torch.distributions.Normal(a_new_sto[:, :, 0], torch.exp(a_new_sto[:, :, 1]))
                log_prob: torch.Tensor = normal_dist.log_prob(a_new_preconv).sum(dim=-1) - torch.log(torch.clamp(corrective_terms, min=1e-5)).sum(dim=-1)
                log_prob = log_prob.unsqueeze(-1) # add the last dimension, given that the sum contracted it

                # ------------------------------------- CLARIFICATION -------------------------------------
                # Each loss is 0.5 * (prediction - target)^2 = 0.5 * MSE(prediction, target)
                # It is not the same the target VALUE of v (in a certain step) and the target NETWORK of v
                # -----------------------------------------------------------------------------------------

                # Target value for each loss
                with torch.no_grad():
                    r = r.to(self.gpu_device)
                    target_v: torch.Tensor = qmin_new - self.temperature * log_prob
                    target_q: torch.Tensor = r + self.discount * vtg_replay_next

                if self.debug:
                    print("LogProbDenBasic:", normal_dist.log_prob(a_new_preconv).sum(dim=-1).mean(), "-log(corrective):", - torch.log(torch.clamp(corrective_terms, min=1e-5)).sum(dim=-1).mean())
                    print("Shapes --> LogProbDen:", log_prob.shape, "Target values:", target_v.shape, target_q.shape)

                # Set the gradients to zero
                optimizer_v.zero_grad()
                optimizer_q1.zero_grad()
                optimizer_q2.zero_grad()
                optimizer_pi.zero_grad()

                # Compute the losses
                J_v: torch.Tensor = 0.5 * self.mse_loss(v_replay, target_v)
                J_q1: torch.Tensor = 0.5 * self.mse_loss(q1_replay, target_q)
                J_q2: torch.Tensor = 0.5 * self.mse_loss(q2_replay, target_q)
                J_pi: torch.Tensor = self.temperature * log_prob.mean() - qmin_new.mean()

                if self.debug:
                    print("V ----> Loss:", f"{J_v.item():.3f}", "Forward:", f"{v_replay.mean().item():.3f}", "Target:", f"{target_v.mean().item():.3f}", "Qmin:", f"{qmin_new.mean().item():.3f}")
                    print("Q1 ---> Loss:", f"{J_q1.item():.3f}", "Forward:", f"{q1_replay.mean().item():.3f}", "Target:", f"{target_q.mean().item():.3f}")
                    print("Q2 ---> Loss:", f"{J_q2.item():.3f}", "Forward:", f"{q2_replay.mean().item():.3f}", "Target:", f"{target_q.mean().item():.3f}")
                    print("Pi ---> Loss:", f"{J_pi.item():.3f}", "Qmin:", f"{qmin_new.mean().item():.3f}", "Alpha:", f"{self.temperature:.3f}", "LogProbDen:", f"{log_prob.mean().item():.3f}")
                    print("Vtg --> Forward:", f"{vtg_replay_next.mean().item():.3f}", "Reward:", f"{r.mean().item():.3f}")

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
                        vtg_params.data.mul_(1 - self.smooth_coeff)
                        vtg_params.data.add_(self.smooth_coeff * v_params.data)

                if not g == 0 and not self.debug:
                    sys.stdout.write("\033[F")
                print(f"Gradient step {g+1}/{self.gradient_steps} done!")

            print("✔ Iteration done!")

        return actor, q1, q2, v, vtg
    
    def train_openai(self, actor: Actor, q1: QNetwork, q1tg: QNetwork, q2: QNetwork, q2tg: QNetwork, list_states: list[torch.Tensor], list_actions: list[torch.Tensor]):
        """
        Begin the training of the SAC algorithm.
        """
        torch.autograd.set_detect_anomaly(True)

        # Optimizers
        optimizer_q1 = optim.Adam(q1.parameters(), lr=q1.lr)
        optimizer_q2 = optim.Adam(q2.parameters(), lr=q2.lr)
        optimizer_pi = optim.Adam(actor.model.parameters(), lr=actor.lr)

        # Loop flags
        done = False
        iteration = 1

        print("Starting training with the OpenAI version of the algorithm...")

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

                    # Do an environment step
                    done, next_states, next_actions = self.do_1_experience(actor, states, actions, agt)

                    # Break if time is up
                    if done:
                        break

                    # Replace the states and actions lists
                    list_states[idx] = next_states
                    list_actions[idx] = next_actions

                # Break if time is up
                if done:
                    break

                if not e == 0 and not self.debug:
                    sys.stdout.write("\033[F")
                print(f"Environment step {e+1}/{self.environment_steps} done!")
            
            # Break if time is up
            if done:
                break

            # Loop over all gradient steps
            for g in range(self.gradient_steps):
                with torch.no_grad():
                    states, actions, a_norm, r, next_states, next_actions = self.tensor_manager.full_squeeze(*self.replay_buffer.sample(self.batch_size))

                if self.debug:
                    print("Shapes of buffer sampling:", states.shape, actions.shape, a_norm.shape, r.shape, next_states.shape, next_actions.shape)

                # Batchify the tensors neccessary for the transformer
                states, actions, next_states, next_actions = self.tensor_manager.batchify(states, actions, next_states, next_actions) # batchify if necessary

                # -----------------------------------------------------------------------------------------

                # Get the stochastic actions again
                new_stochastic_actions = actor(states.to(self.gpu_device), actions.to(self.gpu_device))

                # Select the last stochastic action
                a_new_sto = new_stochastic_actions[:, -1, :, :]

                # Sample and convert the action
                a_new_preconv, _, a_new_norm = actor.model.reparametrization_trick(a_new_sto)

                # Find the minimum of the Q-networks for the replay buffer sample and the new action
                q1_replay: torch.Tensor = q1(states, actions, a_norm)
                q2_replay: torch.Tensor = q2(states, actions, a_norm)
                qmin_new: torch.Tensor = torch.min(q1(states, actions, a_new_norm), q2(states, actions, a_new_norm))

                k = 1 / self.scaling_factor
                corrective_terms = k / torch.cosh(a_new_preconv / self.scaling_factor)**2 # 1 - tanh^2 = sech^2 = 1 / cosh^2
                normal_dist = torch.distributions.Normal(a_new_sto[:, :, 0], torch.exp(a_new_sto[:, :, 1]))
                log_prob: torch.Tensor = normal_dist.log_prob(a_new_preconv).sum(dim=-1) - torch.log(torch.clamp(corrective_terms, min=1e-5)).sum(dim=-1)
                log_prob = log_prob.unsqueeze(-1) # add the last dimension, given that the sum contracted it

                # -----------------------------------------------------------------------------------------

                # Get the stochastic actions again
                next_new_stochastic_actions = actor(next_states.to(self.gpu_device), next_actions.to(self.gpu_device))

                # Select the last stochastic action
                a_next_new_sto = next_new_stochastic_actions[:, -1, :, :]

                # Sample and convert the action
                a_next_new_preconv, _, a_next_new_norm = actor.model.reparametrization_trick(a_next_new_sto)

                # Find the minimum of the Q-networks for the replay buffer sample and the new action
                qtgmin_next_new: torch.Tensor = torch.min(q1tg(next_states, next_actions, a_next_new_norm), q2tg(next_states, next_actions, a_next_new_norm))

                k = 1 / self.scaling_factor
                corrective_terms = k / torch.cosh(a_next_new_preconv / self.scaling_factor)**2 # 1 - tanh^2 = sech^2 = 1 / cosh^2
                normal_dist = torch.distributions.Normal(a_next_new_sto[:, :, 0], torch.exp(a_next_new_sto[:, :, 1]))
                log_prob_next: torch.Tensor = normal_dist.log_prob(a_next_new_preconv).sum(dim=-1) - torch.log(torch.clamp(corrective_terms, min=1e-5)).sum(dim=-1)
                log_prob_next = log_prob_next.unsqueeze(-1) # add the last dimension, given that the sum contracted it


                # ------------------------------------- CLARIFICATION -------------------------------------
                # Each loss is 0.5 * (prediction - target)^2 = 0.5 * MSE(prediction, target)
                # It is not the same the target VALUE of v (in a certain step) and the target NETWORK of v
                # -----------------------------------------------------------------------------------------

                # Target value for each loss
                with torch.no_grad():
                    target_q: torch.Tensor = r.to(self.gpu_device) + self.discount * (qtgmin_next_new - self.temperature * log_prob_next)

                if self.debug:
                    print("Current state --> LogProbDenBasic:", normal_dist.log_prob(a_new_preconv).sum(dim=-1).mean(), "-log(corrective):", - torch.log(torch.clamp(corrective_terms, min=1e-5)).sum(dim=-1).mean())
                    print("Next state --> LogProbDenBasic:", normal_dist.log_prob(a_next_new_preconv).sum(dim=-1).mean(), "-log(corrective):", - torch.log(torch.clamp(corrective_terms, min=1e-5)).sum(dim=-1).mean())
                    print("Shapes --> Current and next LogProbDen:", log_prob.shape, log_prob_next.shape, "Q-network targets:", target_q.shape)

                # Set the gradients to zero
                optimizer_q1.zero_grad()
                optimizer_q2.zero_grad()
                optimizer_pi.zero_grad()

                # Compute the losses
                J_q1: torch.Tensor = 0.5 * self.mse_loss(q1_replay, target_q)
                J_q2: torch.Tensor = 0.5 * self.mse_loss(q2_replay, target_q)
                J_pi: torch.Tensor = self.temperature * log_prob.mean() - qmin_new.mean()

                if self.debug:
                    print("Q1 ---> Loss:", f"{J_q1.item():.3f}", "Forward:", f"{q1_replay.mean().item():.3f}", "Target:", f"{target_q.mean().item():.3f}")
                    print("Q2 ---> Loss:", f"{J_q2.item():.3f}", "Forward:", f"{q2_replay.mean().item():.3f}", "Target:", f"{target_q.mean().item():.3f}")
                    print("Pi ---> Loss:", f"{J_pi.item():.3f}", "Qmin:", f"{qmin_new.mean().item():.3f}", "Alpha:", f"{self.temperature:.3f}", "LogProbDen:", f"{log_prob.mean().item():.3f}")
                    print("Qtgmin next --> Forward:", f"{qtgmin_next_new.mean().item():.3f}", "Reward:", f"{r.mean().item():.3f}")

                # Store the losses
                self.losses["q1"].append(J_q1.item())
                self.losses["q2"].append(J_q2.item())
                self.losses["pi"].append(J_pi.item())

                # Backpropagate
                J_q1.backward(retain_graph=True)
                J_q2.backward(retain_graph=True)
                J_pi.backward(retain_graph=True)

                # Optimize parameters
                optimizer_q1.step()
                optimizer_q2.step()
                optimizer_pi.step()

                # Soft update the target Q-networks
                with torch.no_grad():
                    for q1_params, q1tg_params in zip(q1.parameters(), q1tg.parameters()):
                        q1tg_params.data.mul_(1 - self.smooth_coeff)
                        q1tg_params.data.add_(self.smooth_coeff * q1_params.data)

                    for q2_params, q2tg_params in zip(q2.parameters(), q2tg.parameters()):
                        q2tg_params.data.mul_(1 - self.smooth_coeff)
                        q2tg_params.data.add_(self.smooth_coeff * q2_params.data)

                if not g == 0 and not self.debug:
                    sys.stdout.write("\033[F")
                print(f"Gradient step {g+1}/{self.gradient_steps} done!")

            print("✔ Iteration done!")

        return actor, q1, q1tg, q2, q2tg
    
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
    
    def save_replay_buffer(self):
        """
        Save the replay buffer.
        """
        torch.save(list(self.replay_buffer.storage), f"{self.save_path}/buffer.pth")
        print("Replay buffer saved!")

    def load_parameters(self, models: dict[str, nn.Module]) -> None:
        """
        Load the parameters for every model.
        """
        if not self.load_params:
            return
        for name, model in models.items():
            if os.path.exists(f"{self.save_path}/{name}.pth"):
                print(f"Loading {name} model...")
                model.load_state_dict(torch.load(f"{self.save_path}/{name}.pth", weights_only=True))
            else:
                raise FileNotFoundError(f"Parameters for {name} do not exist.")

    def save_parameters(self, models: dict[str, nn.Module]):
        """
        Save the models with the specified name.
        """
        for name, model in models.items():
            torch.save(model.state_dict(), f"{self.save_path}/{name}.pth")
            print(f"Parameters for {name} saved!")