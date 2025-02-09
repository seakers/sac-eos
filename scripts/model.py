import torch
import torch.nn as nn
import torch.nn.init as init

# from scripts.utils import Reparametrizer

#####################################################
#
# TRANSFORMER ARCHITECTURE. BOTH ENCODER AND DECODER.
#
#####################################################

class FloatEmbedder(nn.Module):
    """"
    Class to embed the float (continuous) values of the states and actions. Child class of nn.Module.
    """
    def __init__(self, input_dim: int, embed_dim: int, dropout: float): # input_dim is state_dim or action_dim
        super(FloatEmbedder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=dropout)
        
        layers = []
        layers.append(nn.Linear(input_dim, self.embed_dim // 4))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.embed_dim // 4, self.embed_dim // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.embed_dim // 2, self.embed_dim))
        layers.append(nn.LayerNorm(self.embed_dim))
        self.embed = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for layer in self.embed:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)
        return x
    
class PositionalEncoder(nn.Module):
    """
    Class to encode the position of the states and actions using the Attention Is All You Need functions. Child class of nn.Module.
    """
    def __init__(self, max_len: int, d_model: int, dropout: float):
        super(PositionalEncoder, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.empty((0, d_model), dtype=torch.float32, requires_grad=True)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0, requires_grad=True)) / d_model))

        for pos in range(max_len):
            sines = torch.sin(pos * div_term)
            cosines = torch.cos(pos * div_term)

            interleaved = torch.stack((sines, cosines), dim=1).flatten()

            pe = torch.cat((pe, interleaved.unsqueeze(0)), dim=0)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.pe = pe

    def forward(self, x):
        x = (x + self.pe[:x.size(0), :])
        x = self.dropout(x)
        return x
    
class SegmentPositionalEncoder(nn.Module):
    """
    Class to encode the position of the states and actions using the a concatenated tensor which represents the sequence position. Child class of nn.Module.
    """
    def __init__(self, max_len: int, d_model: int, encoding_segment_size: int, dropout: float):
        super(SegmentPositionalEncoder, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.encoding_segment_size = encoding_segment_size
        self.embed = nn.Embedding(max_len, encoding_segment_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
        x = torch.cat([x, self.embed(positions)], dim=-1)
        x = self.dropout(x)
        return x

class EOSTransformer(nn.Transformer):
    """
    Class to create a transformer for the Earth Observation Satellite model. Child class of nn.Transformer.
    """
    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout: float,
            activation: str,
            batch_first: bool = True,
            kaiming_init: bool = False
        ):
        super(EOSTransformer, self).__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )
        self.architecture_type = "Transformer"

        if kaiming_init:
            self.override_linears_with_kaiming()

    def override_linears_with_kaiming(self):
        # Encoder
        for layer in self.encoder.layers:
            for module in layer.children():
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    init.zeros_(module.bias)

        # Decoder
        for layer in self.decoder.layers:
            for module in layer.children():
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    init.zeros_(module.bias)

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask
    
class StochasticProjector(nn.Module):
    """
    Class to project the output of the transformer into a stochastic action (twice the size of actions' size). Child class of nn.Module.
    """
    def __init__(self, d_model: int, action_dim: int):
        super(StochasticProjector, self).__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.output_dim = int(2 * self.action_dim)

        layers = []
        layers.append(nn.Linear(self.d_model, self.d_model // 2))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(self.d_model // 2, self.output_dim))
        layers.append(nn.LayerNorm(self.output_dim))
        self.project = nn.Sequential(*layers)

    def init_weights(self):
        for layer in self.project:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
                init.zeros_(layer.bias)

    def forward(self, x):
        x = self.project(x)
        return x
    
class TransformerModelEOS(nn.Module):
    """
    Class to create the Earth Observation Satellite model. Child class of nn.Module. Parts:
        · 1a. embedder for the states
        · 1b. embedder for the actions
        · 2. positional encoder
        · 3. transformer
        · 4. stochastic projector
    """
    def __init__(
            self,
            state_embedder: FloatEmbedder,
            action_embedder: FloatEmbedder,
            pos_encoder: PositionalEncoder,
            transformer: EOSTransformer,
            projector: StochasticProjector,
            a_conversions: torch.tensor
        ):
        super(TransformerModelEOS, self).__init__()
        self.model_type = "Earth Observation Model"
        self.state_embedder = state_embedder
        self.action_embedder = action_embedder
        self.pos_encoder = pos_encoder
        self.transformer = transformer
        self.projector = projector
        self.gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.a_conversions = a_conversions.to(self.gpu_device)
        self.epsilon = torch.distributions.Normal(0, 1)
        self.scaling_factor = torch.sqrt(torch.tensor(1 + torch.e**2, device=self.gpu_device))

        # Check it is a transformer
        if self.transformer.architecture_type != "Transformer":
            raise ValueError("The model is not a transformer")

    def forward(self, states, actions):
        with torch.no_grad():
            # Check if states and actions have the same number of dimensions
            if states.dim() != actions.dim():
                raise ValueError("For cleaner code, the number of dimensions of states and actions should be the same")
            
            # Check whether they are batched or not
            if states.dim() == 1:
                states = states.unsqueeze(0).unsqueeze(0)
                actions = actions.unsqueeze(0).unsqueeze(0)
            elif states.dim() == 2:
                states = states.unsqueeze(0)
                actions = actions.unsqueeze(0)
            
            # Check if the number of states and actions are equal
            if states.shape[1] != actions.shape[1]:
                raise ValueError("The number of states and actions must be equal")
            else:
                seq_len = states.shape[1]

        # Pass the states tensor through the embedding layer
        embedded_states = self.state_embedder(states)

        # Pass the actions tensor through the embedding layer
        embedded_actions = self.action_embedder(actions)

        # Pass the embedded states and actions through the positional encoder
        input_states = self.pos_encoder(embedded_states)
        input_actions = self.pos_encoder(embedded_actions)

        # Set the source and target masks
        mask = self.transformer._generate_square_subsequent_mask(seq_len)

        # Pass the input states and actions through the transformer
        output = self.transformer(input_states, input_actions, src_mask=mask, tgt_mask=mask, memory_mask=mask, src_is_causal=True, tgt_is_causal=True, memory_is_causal=True)

        # Pass the output through the projector
        stochastic_actions = self.projector(output)

        # Group the actions by features with their mean and variance
        stochastic_actions = stochastic_actions.view(-1, seq_len, actions.shape[-1], 2)

        return stochastic_actions
    
    def sample(self, stochastic_actions, batched):
        if batched:
            # Create a sampled actions tensor with the same shape as the stochastic actions tensor
            _, seq_len, action_dim, _ = stochastic_actions.shape
            sampled_actions = torch.empty((0, seq_len, action_dim), requires_grad=True, device=self.gpu_device)

            for batch in stochastic_actions: # for each batch (set of sequences)
                # Create a tensor for the batch
                batch_sampled_actions = torch.empty((0, action_dim), requires_grad=True, device=self.gpu_device)
                for action in batch:
                    # Create a tensor for the sequence
                    seq_sampled_actions = torch.empty((0), requires_grad=True, device=self.gpu_device)
                    for feature in action:
                        mean = feature[0]
                        log_std = feature[1]
                        std = torch.exp(log_std)

                        result = mean + std * self.epsilon.rsample()
                        seq_sampled_actions = torch.cat([seq_sampled_actions, result.unsqueeze(0)], dim=0)
                    
                    batch_sampled_actions = torch.cat([batch_sampled_actions, seq_sampled_actions.unsqueeze(0)], dim=0)
                
                sampled_actions = torch.cat([sampled_actions, batch_sampled_actions.unsqueeze(0)], dim=0)

            return sampled_actions
        else:
            # Create a sampled actions tensor with the same shape as the stochastic actions tensor
            _, action_dim, _ = stochastic_actions.shape
            sampled_actions = torch.empty((0, action_dim), requires_grad=True, device=self.gpu_device)

            for action in stochastic_actions:
                # Create a tensor for the action
                action_sampled_actions = torch.empty((0), requires_grad=True, device=self.gpu_device)
                for feature in action:
                    mean = feature[0]
                    log_std = feature[1]
                    std = torch.exp(log_std)

                    result = mean + std * self.epsilon.rsample()
                    action_sampled_actions = torch.cat([action_sampled_actions, result.unsqueeze(0)], dim=0)

                sampled_actions = torch.cat([sampled_actions, action_sampled_actions.unsqueeze(0)], dim=0)

            return sampled_actions
    
    def convert(self, sampled_actions: torch.tensor, normalized: bool=False):
        if normalized:
            return torch.tanh(sampled_actions / self.scaling_factor)
        else:
            return torch.tanh(sampled_actions / self.scaling_factor) * self.a_conversions
    
    def reparametrization_trick(self, stochastic_actions):
        # Number of dimensions of the tensor
        dim = stochastic_actions.dim()

        if dim == 4:
            # Create the specific output actions tensor
            sampled_actions = self.sample(stochastic_actions, True) # already in a sequence and batched
        elif dim == 3:
            # Create the specific output actions tensor
            sampled_actions = self.sample(stochastic_actions, False)
        elif dim == 2:
            # Create the specific output actions tensor
            sampled_actions = self.sample(stochastic_actions.unsqueeze(0), False).squeeze() # not in a sequence nor batched
        else:
            raise ValueError("The tensor must have 2, 3 or 4 dimensions")
        
        return sampled_actions, self.convert(sampled_actions, normalized=False), self.convert(sampled_actions, normalized=True)  # conversion based on the implementation procedure and the actions dimension
       
#########################################################
#
# MULTI-LAYER PERCEPTRON ARCHITECTURE. SIMPLE AND DENSE.
#
#########################################################

class MLPModelEOS(nn.Module):
    """
    Class to create a Multi-Layer Perceptron model. Child class of nn.Module.
    """
    def __init__(self, state_dim: int, action_dim: int, n_hidden: tuple[int], dropout: float, a_conversions: torch.tensor):
        super(MLPModelEOS, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_hidden = n_hidden
        self.gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.a_conversions = a_conversions.to(self.gpu_device)
        self.epsilon = torch.distributions.Normal(0, 1)
        self.scaling_factor = torch.sqrt(torch.tensor(1 + torch.e**2, device=self.gpu_device))

        layers = []
        layers.append(nn.Linear(state_dim, n_hidden[0]))
        layers.append(nn.ReLU())

        for i in range(len(n_hidden) - 1):
            layers.append(nn.Linear(n_hidden[i], n_hidden[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        
        layers.append(nn.LayerNorm(n_hidden[-1]))
        layers.append(nn.Linear(n_hidden[-1], 2*action_dim))
        self.mlp = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, *args):
        # Rearange the input tensor so that all past states are concatenated
        x = x.view(x.shape[0], -1).unsqueeze(1) # (batch_size, seq_len, state_dim) -> (batch_size, 1, seq_len * state_dim)

        # Check it has 2 dimensions
        if x.dim() != 3:
            raise ValueError("The input tensor must have 3 dimensions: (batch_size, seq_len, state_dim)")
        
        # Check if the input tensor has the right number of sequences
        if x.shape[-1] != self.state_dim:
            # Add zeros to the tensor where x has size (batch_size, seq_len, state_dim)
            x = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], self.state_dim - x.shape[-1]).to(self.gpu_device)], dim=-1)
        
        # Pass the input tensor through the MLP
        stochastic_actions = self.mlp(x)

        # Group the actions by features with their mean and variance
        stochastic_actions = stochastic_actions.view(stochastic_actions.shape[0], stochastic_actions.shape[1], self.action_dim, 2)

        return stochastic_actions

    def sample(self, stochastic_actions, batched):
        if batched:
            # Create a sampled actions tensor with the same shape as the stochastic actions tensor
            _, seq_len, action_dim, _ = stochastic_actions.shape
            sampled_actions = torch.empty((0, seq_len, action_dim), requires_grad=True, device=self.gpu_device)

            for batch in stochastic_actions: # for each batch (set of sequences)
                # Create a tensor for the batch
                batch_sampled_actions = torch.empty((0, action_dim), requires_grad=True, device=self.gpu_device)
                for action in batch:
                    # Create a tensor for the sequence
                    seq_sampled_actions = torch.empty((0), requires_grad=True, device=self.gpu_device)
                    for feature in action:
                        mean = feature[0]
                        log_std = feature[1]
                        std = torch.exp(log_std)

                        result = mean + std * self.epsilon.rsample()
                        seq_sampled_actions = torch.cat([seq_sampled_actions, result.unsqueeze(0)], dim=0)
                    
                    batch_sampled_actions = torch.cat([batch_sampled_actions, seq_sampled_actions.unsqueeze(0)], dim=0)
                
                sampled_actions = torch.cat([sampled_actions, batch_sampled_actions.unsqueeze(0)], dim=0)

            return sampled_actions
        else:
            # Create a sampled actions tensor with the same shape as the stochastic actions tensor
            _, action_dim, _ = stochastic_actions.shape
            sampled_actions = torch.empty((0, action_dim), requires_grad=True, device=self.gpu_device)

            for action in stochastic_actions:
                # Create a tensor for the action
                action_sampled_actions = torch.empty((0), requires_grad=True, device=self.gpu_device)
                for feature in action:
                    mean = feature[0]
                    log_std = feature[1]
                    std = torch.exp(log_std)

                    result = mean + std * self.epsilon.rsample()
                    action_sampled_actions = torch.cat([action_sampled_actions, result.unsqueeze(0)], dim=0)

                sampled_actions = torch.cat([sampled_actions, action_sampled_actions.unsqueeze(0)], dim=0)

            return sampled_actions
    
    def convert(self, sampled_actions: torch.tensor, normalized: bool=False):
        if normalized:
            return torch.tanh(sampled_actions / self.scaling_factor)
        else:
            return torch.tanh(sampled_actions / self.scaling_factor) * self.a_conversions
    
    def reparametrization_trick(self, stochastic_actions):
        # Number of dimensions of the tensor
        dim = stochastic_actions.dim()

        if dim == 4:
            # Create the specific output actions tensor
            sampled_actions = self.sample(stochastic_actions, True) # already in a sequence and batched
        elif dim == 3:
            # Create the specific output actions tensor
            sampled_actions = self.sample(stochastic_actions, False)
        elif dim == 2:
            # Create the specific output actions tensor
            sampled_actions = self.sample(stochastic_actions.unsqueeze(0), False).squeeze() # not in a sequence nor batched
        else:
            raise ValueError("The tensor must have 2, 3 or 4 dimensions")
        
        return sampled_actions, self.convert(sampled_actions, normalized=False), self.convert(sampled_actions, normalized=True)  # conversion based on the implementation procedure and the actions dimension
 