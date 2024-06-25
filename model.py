import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, TransformerEncoder, TransformerEncoderLayer


# %% ====================== GCN Model Building ======================
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        # Store input and output feature dimensions
        self.in_features = in_features
        self.out_features = out_features

        # Create a trainable weight matrix
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        # Optionally create a trainable bias vector
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            # If bias is not needed, register a placeholder parameter (None)
            self.register_parameter("bias", None)

        # Initialize weights and bias using a sensible default method
        self.reset_parameters()

    def reset_parameters(self):
        # Calculate a standard deviation for uniform initialization
        stdv = 1.0 / math.sqrt(self.weight.size(1))  # Fan-in based initialization

        # Initialize weights with random values from a uniform distribution
        self.weight.data.uniform_(-stdv, stdv)

        # Initialize bias (if present) with the same uniform distribution
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # Perform linear transformation of input features
        support = torch.mm(
            input, self.weight
        )  # Linearly transform input features using weights

        # Aggregate transformed features based on graph structure
        output = torch.spmm(
            adj, support
        )  # Apply sparse matrix multiplication with adjacency matrix

        # Optionally add bias and return the output
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        # Create a list to hold multiple Graph Convolution layers
        self.gcn = nn.ModuleList()

        # Store dropout probability and activation function
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Define the number of channels (features) for each layer
        channels = [ninput] + nhid + [noutput]

        # Create Graph Convolution layers with appropriate feature dimensions
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])  # Create a layer
            self.gcn.append(gcn_layer)  # Add the layer to the list

    # Forward pass through the GCN model
    def forward(self, x, adj):
        # Apply multiple Graph Convolution layers with activation
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(
                self.gcn[i](x, adj)
            )  # Apply layer, activation, and pass to next layer

        # Apply dropout for regularization
        x = F.dropout(x, self.dropout, training=self.training)

        # Apply the final Graph Convolution layer without activation
        x = self.gcn[-1](x, adj)

        return x  # Return the final output features


# %% ====================== NodeAttentionMap Model Building ======================
class NodeAttnMap(nn.Module):
    """
    NodeAttnMap class implements a node-level attention mechanism for graph-based data.

    Args:
        in_features (int): Number of input features.
        nhid (int): Number of hidden units.
        use_mask (bool, optional): Whether to use a mask for masked attention. Defaults to False.
    """

    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()

        # Initialize parameters
        self.use_mask = use_mask
        self.out_features = nhid  # Number of output features
        self.W = nn.Parameter(
            torch.empty(size=(in_features, nhid))
        )  # Linear transformation matrix for input
        nn.init.xavier_uniform_(
            self.W.data, gain=1.414
        )  # Initialize weights with Xavier uniform
        self.a = nn.Parameter(
            torch.empty(size=(2 * nhid, 1))
        )  # Parameters for attention mechanism
        nn.init.xavier_uniform_(
            self.a.data, gain=1.414
        )  # Initialize weights with Xavier uniform
        self.leakyrelu = nn.LeakyReLU(0.2)  # Activation function

    def forward(self, X, A):
        """
        Forward pass of the NodeAttnMap module.

        Args:
            X (torch.Tensor): Input features of shape (batch_size, num_nodes, in_features).
            A (torch.Tensor): Adjacency matrix of shape (batch_size, num_nodes, num_nodes).

        Returns:
            torch.Tensor: Attention scores of shape (batch_size, num_nodes, num_nodes).
        """

        # Linear transformation of input features
        Wh = torch.mm(X, self.W)  # (batch_size, num_nodes, nhid)

        # Prepare attention mechanism input
        e = self._prepare_attentional_mechanism_input(
            Wh
        )  # (batch_size, num_nodes, num_nodes)

        # Apply mask if enabled
        if self.use_mask:
            e = torch.where(
                A > 0, e, torch.zeros_like(e)
            )  # Mask attention scores based on adjacency matrix

        # Shift attention scores to 1-2 range and multiply with adjacency matrix
        A = A + 1  # Shift attention scores from 0-1 to 1-2
        e = e * A  # Combine attention with adjacency matrix

        return e  # Return attention scores

    def _prepare_attentional_mechanism_input(self, Wh):
        """
        Prepares the input for the attention mechanism.

        Args:
            Wh (torch.Tensor): Linearly transformed input features.

        Returns:
            torch.Tensor: Attention mechanism input.
        """

        # Split attention parameters
        a1 = self.a[: self.out_features, :]
        a2 = self.a[self.out_features :, :]

        # Calculate attention scores
        Wh1 = torch.matmul(Wh, a1)
        Wh2 = torch.matmul(Wh, a2)
        e = Wh1 + Wh2.T  # Combine attention scores
        e = self.leakyrelu(e)  # Apply activation function

        return e


# %% ====================== UserEmbedding Model Building ======================
class UserEmbeddings(nn.Module):
    # Initialize the embedding layer
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()  # Inherit from the base Module class

        # Create an embedding layer for users
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,  # Number of unique users
            embedding_dim=embedding_dim,  # Dimensionality of each user embedding
        )

    # Generate embeddings for a given batch of user indices
    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)  # Lookup embeddings for each user index
        return embed  # Return the generated embeddings


# %% ====================== TimeEmbedding Model Building ======================
def t2v(tau, f, w, b, w0, b0):
    # Apply a non-linear transformation to tau
    v1 = f(torch.matmul(tau, w) + b)

    # Apply a linear transformation to tau
    v2 = torch.matmul(tau, w0) + b0

    # Concatenate the transformed parts
    return torch.cat([v1, v2], 1)  # Combine results along the feature dimension


# Define a class for Sine Activation layers
class SineActivation(nn.Module):
    # Initialize the layer
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()  # Inherit from the base module class

        # Store output feature dimensions
        self.out_features = out_features

        # Create trainable parameters for multiple sine waves
        self.w0 = nn.Parameter(torch.randn(in_features, 1))  # Base frequency weights
        self.b0 = nn.Parameter(torch.randn(in_features, 1))  # Base frequency biases
        self.w = nn.Parameter(
            torch.randn(in_features, out_features - 1)
        )  # Additional frequency weights
        self.b = nn.Parameter(
            torch.randn(in_features, out_features - 1)
        )  # Additional frequency biases

        # Set the activation function to sine
        self.f = torch.sin

    # Forward pass through the layer
    def forward(self, tau):
        # Apply the transformation to input (tau) using sine activation
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)


# Define a class for Cosine Activation
class CosineActivation(nn.Module):
    # Initialize the layer
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()  # Inherit from the base module class

        # Store the number of output features
        self.out_features = out_features

        # Create trainable parameters for scaling and shifting
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))  # Initial scaling
        self.b0 = nn.parameter.Parameter(
            torch.randn(in_features, 1)
        )  # Initial shifting
        self.w = nn.parameter.Parameter(
            torch.randn(in_features, out_features - 1)
        )  # Further scaling
        self.b = nn.parameter.Parameter(
            torch.randn(in_features, out_features - 1)
        )  # Further shifting

        # Set the activation function to cosine
        self.f = torch.cos

    # Forward pass through the layer
    def forward(self, tau):
        return t2v(
            tau, self.f, self.w, self.b, self.w0, self.b0
        )  # Call an external function for mapping


# Define a class for Time2Vec embedding
class Time2Vec(nn.Module):
    # Initialize the layer
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()  # Inherit from the base module class

        # Choose the activation function based on input
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)  # Use SineActivation layer
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)  # Use CosineActivation layer
        else:
            raise ValueError(
                f"Invalid activation: {activation}"
            )  # Handle invalid activation

    # Forward pass
    def forward(self, x):
        x = self.l1(x)  # Apply the chosen activation layer
        return x


# %% ====================== UserEmbedding Model Building ======================
class CategoryEmbeddings(nn.Module):
    # Initialize the model
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()  # Inherit from the base module class

        # Create an embedding layer for categories
        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,  # Number of unique categories
            embedding_dim=embedding_dim,  # Dimension of each embedding vector
        )

    # Forward pass to generate embeddings
    def forward(self, cat_idx):
        embed = self.cat_embedding(
            cat_idx
        )  # Look up embeddings for input category indices
        return embed  # Return the generated embeddings


# %% ====================== EmbeddingFusion Model Building ======================
class FuseEmbeddings(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(FuseEmbeddings, self).__init__()

        # Combine dimensions of user and POI embeddings
        embed_dim = x_dim + y_dim

        # Create a linear layer to fuse the embeddings
        self.fuse_embed = nn.Linear(
            embed_dim, embed_dim
        )  # Linearly transform concatenated embeddings

        # Add a LeakyReLU activation for non-linearity
        self.leaky_relu = nn.LeakyReLU(
            0.2
        )  # Apply LeakyReLU activation for non-linearity

    def forward(self, xx, yy):
        # Concatenate along the feature dimension
        x = torch.cat((xx, yy), 0)

        # Fuse the concatenated embeddings using the linear layer
        x = self.fuse_embed(x)

        # Apply LeakyReLU activation
        x = self.leaky_relu(x)

        return x  # Return the fused and activated embedding


# %% ====================== Transformer Model Building ======================
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()

        # Create a dropout layer
        self.dropout = torch.nn.Dropout(p=dropout)

        # Create a positional encoding matrix
        pe = torch.zeros(max_len, d_model)  # Initialize with zeros
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # Create position indices
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # Create a division term for scaling
        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # Apply sine wave pattern to even indices
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # Apply cosine wave pattern to odd indices
        pe = pe.unsqueeze(0).transpose(0, 1)  # Reshape for batch compatibility
        self.register_buffer("pe", pe)  # Register as a buffer for efficient access

    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[: x.size(0), :]
        # Apply dropout
        return self.dropout(x)


# Class for a Transformer-based model
class TransformerModel(nn.Module):
    def __init__(
        self, num_pois, num_cats, embed_size, nhead, nhid, nlayers, dropout=0.5
    ):
        super().__init__()

        # Transformer components
        self.pos_encoder = PositionalEncoding(
            embed_size, dropout
        )  # Positional encoding for input sequences
        encoder_layer = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, nlayers
        )  # Transformer encoder stack

        self.embed_size = embed_size
        # Decoders for different outputs
        self.decoder_poi = nn.Linear(
            embed_size, num_pois
        )  # Decode POI (Point of Interest)
        self.decoder_time = nn.Linear(embed_size, 1)  # Decode time
        self.decoder_cat = nn.Linear(embed_size, num_cats)  # Decode category

        # Initialize weights
        self.init_weights()

    # Generate a mask to prevent attending to subsequent positions
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    # Initialize weights (only for decoder_poi here)
    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    # Forward pass
    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)  # Scale input embeddings
        src = self.pos_encoder(src)  # Add positional encoding
        x = self.transformer_encoder(src, src_mask)  # Pass through transformer encoder
        out_poi = self.decoder_poi(x)  # Decode POI
        out_time = self.decoder_time(x)  # Decode time
        out_cat = self.decoder_cat(x)  # Decode category
        return out_poi, out_time, out_cat
