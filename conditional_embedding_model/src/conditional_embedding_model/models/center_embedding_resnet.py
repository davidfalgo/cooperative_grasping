# Reorganized from train/TrainNS_models.py
# Original author:
# Description: Model definitions for center and context embedding variants (v0–v5) using GRU, ResNet18, and LayerNorm

import os
import time
import yaml
import torch
import wandb
import pickle
import itertools
import numpy as np
from torch import nn
from tqdm import tqdm
from torchinfo import summary
from torchvision import models
import torch.nn.functional as F
# from TrainNS_dataloader import *
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from conditional_embedding_model.data import *
from conditional_embedding_model.utils import *

############################################################################################################
######################################## Model v0 simple ##################################################
############################################################################################################

class CenterEmbeddingResidual(nn.Module):
    def __init__(self, input_dim, embedding_dim, feature_structure):
        super(CenterEmbeddingResidual, self).__init__()
        self.feature_structure = feature_structure
        self.features_flatten_shape = self.feature_structure['features_structured']
        self.features_shape = self.feature_structure['features_shape']
        self.split_sizes = [self.features_flatten_shape[key] for key in ['map', 'object_footprint', 'grasping_approach']]

        # Vector Center embedding
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        # Map embedding ()

        # grasp embedding
        self.grasping_gru = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        # center embedding
        self.center_embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x, feature):
        map_feature, object_footprint, grasping_approach = feature.split(self.split_sizes, dim=1) # TODO: FINISH AND TEST
        grasping_approach = grasping_approach.reshape((-1,self.features_shape['grasping_approach'][0], self.features_shape['grasping_approach'][1])).float()

        # Get the center vector
        x = x.long() # FIXME REMOVE BEFORE FLIGHT
        x_expanded = x.unsqueeze(-1).expand(-1, -1, self.features_shape['grasping_approach'][1])
        center = torch.gather(grasping_approach, 1, x_expanded).squeeze(1).float() # [batch, 1, 4] --> [batch, 4]

        center = self.embedder(center) # [batch, 4] --> [batch, 128]
        grasping_embedding = self.embedder(grasping_approach) # [batch, 150, 4] --> [batch, 150, 128]
        grasping_embedding_gru, _ = self.grasping_gru(grasping_embedding) # [batch, 150, 128] --> [batch, 150, 128]
        grasping_embedding_gru = grasping_embedding_gru + grasping_embedding
        grasping_embedding_gru = torch.mean(grasping_embedding_gru, dim=1) # [batch, 128]

        embedding = torch.cat([center, grasping_embedding_gru], dim=1) # [batch, 256]
        # embedding = center + grasping_embedding_gru # [batch, 128]

        return self.center_embedding(embedding)

class ContextEmbeddingResidual(nn.Module):
    def __init__(self, input_dim, embedding_dim, feature_structure):
        super(ContextEmbeddingResidual, self).__init__()
        self.feature_structure = feature_structure
        self.features_flatten_shape = self.feature_structure['features_structured']
        self.features_shape = self.feature_structure['features_shape']
        self.split_sizes = [self.features_flatten_shape[key] for key in ['map', 'object_footprint', 'grasping_approach']]

        # Vector Context embedding
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        # Map embedding ()

        # grasp embedding
        self.grasping_gru = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)#, dropout=0.3)

        # context embedding
        self.context_embedding_gru = nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)#, dropout=0.3)
        self.context_embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x, feature):
        map_feature, object_footprint, grasping_approach = feature.split(self.split_sizes, dim=1)
        grasping_approach = grasping_approach.reshape((-1,self.features_shape['grasping_approach'][0], self.features_shape['grasping_approach'][1])).float()

        x = x.long() # FIXME REMOVE BEFORE FLIGHT
        x_expanded = x.unsqueeze(-1).expand(-1, -1, 4)
        # Get the context tensor - contex shape [batch, max_length, 4]
        context = torch.gather(grasping_approach, 1, x_expanded)
        context = self.embedder(context) # [batch, max_length, 4] --> [batch, max_length, 128]

        grasping_embedding = self.embedder(grasping_approach) # [batch, 150, 4] --> [batch, 150, 128]
        grasping_embedding_gru, _ = self.grasping_gru(grasping_embedding)
        grasping_embedding_gru = grasping_embedding_gru + grasping_embedding
        grasping_embedding_gru = torch.mean(grasping_embedding_gru, dim=1).unsqueeze(1).expand(-1, context.shape[1], -1) # [batch, 1, 128]

        embedding = torch.cat([context, grasping_embedding_gru], dim=2) # [batch, max_length, 256]
        # embedding = context + grasping_embedding_gru
        embedding_context, _ = self.context_embedding_gru(embedding) # [batch, max_length, 256] --> [batch, max_length, 256]
        # embedding_context = embedding_context + embedding # Residual connection
        embedding_context = self.context_embedding(embedding_context)
        return embedding_context

############################################################################################################
##############################@@########## Model with map ##################################################
############################################################################################################

class CenterEmbeddingResNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, feature_structure):
        super(CenterEmbeddingResNet, self).__init__()
        self.feature_structure = feature_structure
        self.features_flatten_shape = self.feature_structure['features_structured']
        self.features_shape = self.feature_structure['features_shape']
        self.split_sizes = [self.features_flatten_shape[key] for key in ['map', 'object_footprint', 'grasping_approach']]

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize(224),  # Resize to 224x224 antialias=True
            transforms.Grayscale(num_output_channels=3),  # Convert to three-channel grayscale
        ])

        # Vector Center embedding
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        # Map embedding ()
        self.image_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Freeze the layers except the last one
        for param in self.image_resnet.parameters():
            param.requires_grad = False

        # Change the last layer
        self.image_resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # grasp embedding
        self.grasping_gru = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        # center embedding
        self.center_embedding = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x, feature):
        map_feature, object_footprint, grasping_approach = feature.split(self.split_sizes, dim=1) # TODO: FINISH AND TEST
        map_image = map_feature.reshape((-1, self.features_shape['map'][0], self.features_shape['map'][1])).unsqueeze(1)
        map_image = self.transform(map_image)  # Now shape [batch_size, 3, 224, 224]
        map_image = map_image.repeat(1, 3, 1, 1).float()  # Now shape [batch_size, 3, 90, 90]
        grasping_approach = grasping_approach.reshape((-1,self.features_shape['grasping_approach'][0], self.features_shape['grasping_approach'][1])).float()

        # Get the center vector
        x = x.long() # FIXME REMOVE BEFORE FLIGHT
        x_expanded = x.unsqueeze(-1).expand(-1, -1, self.features_shape['grasping_approach'][1])
        center = torch.gather(grasping_approach, 1, x_expanded).squeeze(1).float() # [batch, 1, 4] --> [batch, 4]

        center = self.embedder(center) # [batch, 4] --> [batch, 128]
        grasping_embedding = self.embedder(grasping_approach) # [batch, 150, 4] --> [batch, 150, 128]
        grasping_embedding_gru, _ = self.grasping_gru(grasping_embedding) # [batch, 150, 128] --> [batch, 150, 128]
        grasping_embedding_gru = grasping_embedding_gru + grasping_embedding
        grasping_embedding_gru = torch.mean(grasping_embedding_gru, dim=1) # [batch, 128]

        # process the map image
        map_embedding = self.image_resnet(map_image) # [batch, 128]

        embedding = torch.cat([center, grasping_embedding_gru, map_embedding], dim=1) # [batch, 384]
        # embedding = center + grasping_embedding_gru # [batch, 128]

        return self.center_embedding(embedding)

class ContextEmbeddingResNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, feature_structure):
        super(ContextEmbeddingResNet, self).__init__()
        self.feature_structure = feature_structure
        self.features_flatten_shape = self.feature_structure['features_structured']
        self.features_shape = self.feature_structure['features_shape']
        self.split_sizes = [self.features_flatten_shape[key] for key in ['map', 'object_footprint', 'grasping_approach']]

        # Vector Context embedding
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize(224),  # Resize to 224x224 , antialias=True
            transforms.Grayscale(num_output_channels=3),  # Convert to three-channel grayscale
        ])

        # Map embedding ()
        self.image_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Freeze the layers except the last one
        for param in self.image_resnet.parameters():
            param.requires_grad = False

        # Change the last layer
        self.image_resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # grasp embedding
        self.grasping_gru = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)#, dropout=0.3)

        # context embedding
        self.context_embedding_gru = nn.GRU(input_size=384, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)#, dropout=0.3)
        self.context_embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x, feature):
        map_feature, object_footprint, grasping_approach = feature.split(self.split_sizes, dim=1)
        map_image = map_feature.reshape((-1, self.features_shape['map'][0], self.features_shape['map'][1])).unsqueeze(1)
        map_image = self.transform(map_image)  # Now shape [batch_size, 3, 224, 224]
        map_image = map_image.repeat(1, 3, 1, 1).float()  # Now shape [batch_size, 3, 90, 90]
        grasping_approach = grasping_approach.reshape((-1,self.features_shape['grasping_approach'][0], self.features_shape['grasping_approach'][1])).float()

        x = x.long() # FIXME REMOVE BEFORE FLIGHT
        x_expanded = x.unsqueeze(-1).expand(-1, -1, 4)
        # Get the context tensor - contex shape [batch, max_length, 4]
        context = torch.gather(grasping_approach, 1, x_expanded)
        context = self.embedder(context) # [batch, max_length, 4] --> [batch, max_length, 128]

        grasping_embedding = self.embedder(grasping_approach) # [batch, 150, 4] --> [batch, 150, 128]
        grasping_embedding_gru, _ = self.grasping_gru(grasping_embedding)
        grasping_embedding_gru = grasping_embedding_gru + grasping_embedding
        grasping_embedding_gru = torch.mean(grasping_embedding_gru, dim=1).unsqueeze(1).expand(-1, context.shape[1], -1) # [batch, 1, 128]

        # process the map image
        map_embedding = self.image_resnet(map_image).unsqueeze(1).expand(-1, context.shape[1], -1) # [batch, 1, 128]

        embedding = torch.cat([context, grasping_embedding_gru, map_embedding], dim=2) # [batch, max_length, 384]
        # embedding = context + grasping_embedding_gru
        embedding_context, _ = self.context_embedding_gru(embedding) # [batch, max_length, 256] --> [batch, max_length, 256]
        # embedding_context = embedding_context + embedding # Residual connection
        embedding_context = self.context_embedding(embedding_context)
        return embedding_context

############################################################################################################
####################################### Model v1 Transformer ###############################################
############################################################################################################

class MaskedSelfAttention(torch.nn.Module):
    """
    Pytorch module for a self attention layer.
    This layer is used in the MultiHeadedSelfAttention module.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, head_dimension)
    """

    def __init__(self, embedding_dimension, head_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        self.query_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.key_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.value_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, mask):
        """
        Compute the self attention.

        x dimension is: (batch_size, sequence_length, embedding_dimension)
        output dimension is: (batch_size, sequence_length, head_dimension)
        mask dimension is: (batch_size, sequence_length)

        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """

        # x dimensions are: (batch_size, sequence_length, embedding_dimension)
        # query, key, value dimensions are: (batch_size, sequence_length, head_dimension)
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        # Calculate the attention weights.
        # attention_weights dimensions are: (batch_size, sequence_length, sequence_length)
        attention_weights = torch.matmul(query, key.transpose(-2, -1))

        # Scale the attention weights.
        attention_weights = attention_weights / np.sqrt(self.head_dimension)

        # Apply the mask to the attention weights, by setting the masked tokens to a very low value.
        # This will make the softmax output 0 for these values.
        mask = mask.reshape(attention_weights.shape[0], 1, attention_weights.shape[2])
        attention_weights = attention_weights.masked_fill(mask == 0, -1e9)

        # Softmax makes sure all scores are between 0 and 1 and the sum of scores is 1.
        # attention_scores dimensions are: (batch_size, sequence_length, sequence_length)
        attention_scores = self.softmax(attention_weights)

        # The attention scores are multiplied by the value
        # Values of tokens with high attention score get highlighted because they are multiplied by a larger number,
        # and tokens with low attention score get drowned out because they are multiplied by a smaller number.
        # Output dimensions are: (batch_size, sequence_length, head_dimension)
        return torch.bmm(attention_scores, value)

class MaskedMultiHeadedSelfAttention(torch.nn.Module):
    """
    Pytorch module for a multi head attention layer.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(self, embedding_dimension, number_of_heads):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = embedding_dimension // number_of_heads
        self.number_of_heads = number_of_heads

        # Create the self attention modules
        self.self_attentions = torch.nn.ModuleList(
            [MaskedSelfAttention(embedding_dimension, self.head_dimension) for _ in range(number_of_heads)])

        # Create a linear layer to combine the outputs of the self attention modules
        self.output_layer = torch.nn.Linear(number_of_heads * self.head_dimension, embedding_dimension)

    def forward(self, x, mask):
        """
        Compute the multi head attention.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        mask dimensions are: (batch_size, sequence_length)
        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """
        # Compute the self attention for each head
        # self_attention_outputs dimensions are:
        # (number_of_heads, batch_size, sequence_length, head_dimension)
        self_attention_outputs = [self_attention(x, mask) for self_attention in self.self_attentions]

        # Concatenate the self attention outputs
        # self_attention_outputs_concatenated dimensions are:
        # (batch_size, sequence_length, number_of_heads * head_dimension)
        concatenated_self_attention_outputs = torch.cat(self_attention_outputs, dim=2)

        # Apply the output layer to the concatenated self attention outputs
        # output dimensions are: (batch_size, sequence_length, embedding_dimension)
        return self.output_layer(concatenated_self_attention_outputs)

class FeedForward(torch.nn.Module):
    """
    Pytorch module for a feed forward layer.

    A feed forward layer is a fully connected layer with a ReLU activation function in between.
    """

    def __init__(self, embedding_dimension, feed_forward_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.feed_forward_dimension = feed_forward_dimension
        self.linear_1 = torch.nn.Linear(embedding_dimension, feed_forward_dimension)
        self.linear_2 = torch.nn.Linear(feed_forward_dimension, embedding_dimension)

    def forward(self, x):
        """
        Compute the feed forward layer.
        """
        return self.linear_2(torch.relu(self.linear_1(x)))

class DecoderLayer(torch.nn.Module):
    """
    Pytorch module for an encoder layer.

    An encoder layer consists of a multi-headed self attention layer, a feed forward layer and dropout.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(
            self,
            embedding_dimension,
            number_of_heads,
            feed_forward_dimension,
            dropout_rate
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate

        self.multi_headed_self_attention = MaskedMultiHeadedSelfAttention(embedding_dimension, number_of_heads)
        self.feed_forward = FeedForward(embedding_dimension, feed_forward_dimension)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer_normalization_1 = torch.nn.LayerNorm(embedding_dimension)
        self.layer_normalization_2 = torch.nn.LayerNorm(embedding_dimension)

    def forward(self, x, mask):
        """
        Compute the encoder layer.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        mask dimensions are: (batch_size, sequence_length)
        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """

        # Layer normalization 1
        normalized_x = self.layer_normalization_1(x)

        # Multi headed self attention
        attention_output = self.multi_headed_self_attention(normalized_x, mask)

        # Residual output
        residual_output = x + attention_output

        # Layer normalization 2
        normalized_residual_output = self.layer_normalization_2(residual_output)

        # Feed forward
        feed_forward_output = self.feed_forward(normalized_residual_output)

        # Dropout, only when training.
        if self.training:
            feed_forward_output = self.dropout(feed_forward_output)

        # Residual output
        return residual_output + feed_forward_output

class DecoderStack(torch.nn.Module):
    """
    Pytorch module for a stack of decoders.
    """

    def __init__(
            self,
            embedding_dimension,
            number_of_layers,
            number_of_heads,
            feed_forward_dimension,
            dropout_rate,
            max_sequence_length
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length

        # Create the encoder layers
        self.encoder_layers = torch.nn.ModuleList(
            [DecoderLayer(embedding_dimension, number_of_heads, feed_forward_dimension, dropout_rate) for _ in
             range(number_of_layers)])

    def forward(self, x, mask):
        decoder_outputs = x
        for decoder_layer in self.encoder_layers:
            decoder_outputs = decoder_layer(decoder_outputs, mask)

        return decoder_outputs

class LanguageModel(torch.nn.Module):
    """
    Pytorch module for a language model.
    """

    def __init__(
            self,
            number_of_tokens,  # The number of tokens in the vocabulary
            max_sequence_length=512,  # The maximum sequence length to use for attention
            embedding_dimension=512,  # The dimension of the token embeddings
            number_of_layers=6,  # The number of decoder layers to use
            number_of_heads=4,  # The number of attention heads to use
            feed_forward_dimension=None,  # The dimension of the feed forward layer
            dropout_rate=0.1  # The dropout rate to use
    ):
        super().__init__()
        self.number_of_tokens = number_of_tokens
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads

        if feed_forward_dimension is None:
            # GPT-2 paper uses 4 * embedding_dimension for the feed forward dimension
            self.feed_forward_dimension = embedding_dimension * 4
        else:
            self.feed_forward_dimension = feed_forward_dimension

        self.dropout_rate = dropout_rate

        # Create the token embedding layer
        self.token_embedding = TokenEmbedding(embedding_dimension, number_of_tokens)

        # Create the positional encoding layer
        self.positional_encoding = PositionalEncoding(embedding_dimension, max_sequence_length)

        # Create the normalization layer
        self.layer_normalization = torch.nn.LayerNorm(embedding_dimension)

        # Create the decoder stack
        self.decoder = DecoderStack(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_layers,
            number_of_heads=number_of_heads,
            feed_forward_dimension=self.feed_forward_dimension,
            dropout_rate=dropout_rate,
            max_sequence_length=max_sequence_length
        )

        # Create the language model head
        self.lm_head = LMHead(embedding_dimension, number_of_tokens)

    def forward(self, x, mask):
        # Compute the token embeddings
        # token_embeddings dimensions are: (batch_size, sequence_length, embedding_dimension)
        token_embeddings = self.token_embedding(x)

        # Compute the positional encoding
        # positional_encoding dimensions are: (batch_size, sequence_length, embedding_dimension)
        positional_encoding = self.positional_encoding(token_embeddings)

        # Post embedding layer normalization
        positional_encoding_normalized = self.layer_normalization(positional_encoding)

        decoder_outputs = self.decoder(positional_encoding_normalized, mask)
        lm_head_outputs = self.lm_head(decoder_outputs)

        return lm_head_outputs

###############################################################################################################
##############################@@########## Model v2 with map ##################################################
###############################################################################################################

class CenterEmbeddingResNetv2(nn.Module):
    def __init__(self, input_dim, embedding_dim, feature_structure, dropout=0.1):
        super(CenterEmbeddingResNetv2, self).__init__()
        self.feature_structure = feature_structure
        self.features_flatten_shape = self.feature_structure['features_structured']
        self.features_shape = self.feature_structure['features_shape']
        self.split_sizes = [self.features_flatten_shape[key] for key in ['map', 'object_footprint', 'grasping_approach']]

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize(224),  # Resize to 224x224 antialias=True
            transforms.Grayscale(num_output_channels=3),  # Convert to three-channel grayscale
        ])

        # Vector Center embedding
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            # nn.BatchNorm1d(128)     # regularization
        )

        # Object footprint embedding
        self.footprint_embedding = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            # nn.BatchNorm1d(16)      # regularization
        )

        # Map embedding ()
        # self.image_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_resnet = models.resnet18(pretrained=True)
        # Freeze the layers except the last one
        for param in self.image_resnet.parameters():
            param.requires_grad = False

        # Change the last layer
        self.image_resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # grasp embedding
        self.grasping_gru = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        # object footprint embedding
        self.footprint_gru = nn.GRU(input_size=16, hidden_size=8, num_layers=2, batch_first=True, bidirectional=True)

        # center embedding
        self.center_embedding = nn.Sequential(
            nn.Linear(400, 256),
            nn.ReLU(),
            nn.Dropout(dropout),        # regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),        # regularization
            # nn.BatchNorm1d(128),    # regularization
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x, feature):
        # Get the variables and convert to the correct format
        map_feature, object_footprint, grasping_approach = feature.split(self.split_sizes, dim=1) # TODO: FINISH AND TEST
        map_image = map_feature.reshape((-1, self.features_shape['map'][0], self.features_shape['map'][1])).unsqueeze(1)
        map_image = map_image.repeat(1, 3, 1, 1).float()  # Now shape [batch_size, 3, 90, 90]
        map_image = self.transform(map_image)  # Now shape [batch_size, 3, 224, 224]

        grasping_approach = grasping_approach.reshape((-1,self.features_shape['grasping_approach'][0], self.features_shape['grasping_approach'][1])).float()
        object_footprint = object_footprint.reshape((-1,self.features_shape['object_footprint'][0], self.features_shape['object_footprint'][1])).float()

        # Get the center vector
        x = x.long() # FIXME REMOVE BEFORE FLIGHT
        x_expanded = x.unsqueeze(-1).expand(-1, -1, self.features_shape['grasping_approach'][1])
        center = torch.gather(grasping_approach, 1, x_expanded).squeeze(1).float() # [batch, 1, 4] --> [batch, 4]

        center = self.embedder(center) # [batch, 4] --> [batch, 128]
        grasping_embedding = self.embedder(grasping_approach) # [batch, 150, 4] --> [batch, 150, 128]
        grasping_embedding_gru, _ = self.grasping_gru(grasping_embedding) # [batch, 150, 128] --> [batch, 150, 128]
        grasping_embedding_gru = grasping_embedding_gru + grasping_embedding
        grasping_embedding_gru = torch.mean(grasping_embedding_gru, dim=1) # [batch, 128]

        # process the footprint
        footprint_embedding = self.footprint_embedding(object_footprint) # [batch, 4, 2] --> [batch, 4, 16]
        footprint_embedding_gru, _ = self.footprint_gru(footprint_embedding) # [batch, 4, 16] --> [batch, 4, 16]
        footprint_embedding_gru = footprint_embedding_gru + footprint_embedding
        footprint_embedding_gru = torch.mean(footprint_embedding_gru, dim=1) # [batch, 16]

        # process the map image
        map_embedding = self.image_resnet(map_image) # [batch, 128]

        embedding = torch.cat([center, grasping_embedding_gru, map_embedding, footprint_embedding_gru], dim=1) # [batch, 400]

        return self.center_embedding(embedding)

class ContextEmbeddingResNetv2(nn.Module):
    def __init__(self, input_dim, embedding_dim, feature_structure, dropout=0.1):
        super(ContextEmbeddingResNetv2, self).__init__()
        self.feature_structure = feature_structure
        self.features_flatten_shape = self.feature_structure['features_structured']
        self.features_shape = self.feature_structure['features_shape']
        self.split_sizes = [self.features_flatten_shape[key] for key in ['map', 'object_footprint', 'grasping_approach']]

        # Vector Context embedding
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            # nn.BatchNorm1d(128)     # regularization
        )

        # Object footprint embedding
        self.footprint_embedding = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            # nn.BatchNorm1d(16)      # regularization
        )

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize(224),  # Resize to 224x224 , antialias=True
            transforms.Grayscale(num_output_channels=3),  # Convert to three-channel grayscale
        ])

        # Map embedding ()
        # self.image_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_resnet = models.resnet18(pretrained=True)
        # Freeze the layers except the last one
        for param in self.image_resnet.parameters():
            param.requires_grad = False

        # Change the last layer
        self.image_resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # grasp embedding
        self.grasping_gru = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)#, dropout=0.3)
        # object footprint embedding
        self.footprint_gru = nn.GRU(input_size=16, hidden_size=8, num_layers=2, batch_first=True, bidirectional=True)

        # context embedding
        self.context_embedding_gru = nn.GRU(input_size=400, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)#, dropout=0.3)
        self.context_embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.BatchNorm1d(128),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x, feature):
        # Get the variables and convert to the correct format
        map_feature, object_footprint, grasping_approach = feature.split(self.split_sizes, dim=1)
        map_image = map_feature.reshape((-1, self.features_shape['map'][0], self.features_shape['map'][1])).unsqueeze(1)
        map_image = map_image.repeat(1, 3, 1, 1).float()  # Now shape [batch_size, 3, 90, 90]
        map_image = self.transform(map_image)  # Now shape [batch_size, 3, 224, 224]

        grasping_approach = grasping_approach.reshape((-1,self.features_shape['grasping_approach'][0], self.features_shape['grasping_approach'][1])).float()
        object_footprint = object_footprint.reshape((-1,self.features_shape['object_footprint'][0], self.features_shape['object_footprint'][1])).float()

        x = x.long() # FIXME REMOVE BEFORE FLIGHT
        x_expanded = x.unsqueeze(-1).expand(-1, -1, 4)
        # Get the context tensor - contex shape [batch, max_length, 4]
        context = torch.gather(grasping_approach, 1, x_expanded)
        context = self.embedder(context) # [batch, max_length, 4] --> [batch, max_length, 128]

        grasping_embedding = self.embedder(grasping_approach) # [batch, 150, 4] --> [batch, 150, 128]
        grasping_embedding_gru, _ = self.grasping_gru(grasping_embedding)
        grasping_embedding_gru = grasping_embedding_gru + grasping_embedding
        grasping_embedding_gru = torch.mean(grasping_embedding_gru, dim=1).unsqueeze(1).expand(-1, context.shape[1], -1) # [batch, 1, 128]

        # process the footprint
        footprint_embedding = self.footprint_embedding(object_footprint) # [batch, 4, 2] --> [batch, 4, 16]
        footprint_embedding_gru, _ = self.footprint_gru(footprint_embedding) # [batch, 4, 16] --> [batch, 4, 16]
        footprint_embedding_gru = footprint_embedding_gru + footprint_embedding
        footprint_embedding_gru = torch.mean(footprint_embedding_gru, dim=1).unsqueeze(1).expand(-1, context.shape[1], -1) # [batch, 1, 16]

        # process the map image
        map_embedding = self.image_resnet(map_image).unsqueeze(1).expand(-1, context.shape[1], -1) # [batch, 1, 128]

        embedding = torch.cat([context, grasping_embedding_gru, map_embedding, footprint_embedding_gru], dim=2) # [batch, max_length, 400]
        # embedding = context + grasping_embedding_gru
        embedding_context, _ = self.context_embedding_gru(embedding) # [batch, max_length, 400] --> [batch, max_length, 256]
        # embedding_context = embedding_context + embedding # Residual connection
        embedding_context = self.context_embedding(embedding_context)
        return embedding_context

###############################################################################################################
##############################@@########## Model v3 with map ##################################################
###############################################################################################################

class CenterEmbeddingResNetv3(nn.Module):
    def __init__(self, input_dim, embedding_dim, feature_structure, dropout=0.1):
        super(CenterEmbeddingResNetv3, self).__init__()
        self.feature_structure = feature_structure
        self.features_flatten_shape = self.feature_structure['features_structured']
        self.features_shape = self.feature_structure['features_shape']
        self.split_sizes = [self.features_flatten_shape[key] for key in ['map', 'object_footprint', 'grasping_approach']]

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize(224),  # Resize to 224x224 antialias=True
            transforms.Grayscale(num_output_channels=3),  # Convert to three-channel grayscale
            transforms.Normalize(                           # ImageNet statistics
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

        # Vector Center embedding
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            # nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            # nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            # nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        # Object footprint embedding
        self.footprint_embedding = nn.Sequential(
            nn.Linear(2, 8),
            # nn.LayerNorm(8),
            nn.ReLU(),
            nn.Linear(8, 16),
            # nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            # nn.BatchNorm1d(16)      # regularization
        )

        # Map embedding ()
        # self.image_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_resnet = models.resnet18(pretrained=True)
        # Freeze the layers except the last one
        for name, param in self.image_resnet.named_parameters():
            # keep early layers frozen, fine-tune high-level features
            param.requires_grad = ("layer4" in name) or ("fc" in name)

        # Change the last layer
        self.image_resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            # nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # grasp embedding
        self.grasping_gru = nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)#, dropout=dropout)

        # object footprint embedding
        self.footprint_gru = nn.GRU(input_size=32, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)#, dropout=dropout)

        # center embedding
        self.center_embedding = nn.Sequential(
            nn.Linear(256+256+128+32, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),        # regularization
            nn.Linear(256, 128),
            # nn.LayerNorm(128),
            nn.ReLU(),
            # nn.BatchNorm1d(128),
            nn.Dropout(dropout),        # regularization
            # nn.BatchNorm1d(128),    # regularization
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x, feature):
        # Get the variables and convert to the correct format
        map_feature, object_footprint, grasping_approach = feature.split(self.split_sizes, dim=1) # TODO: FINISH AND TEST
        map_image = map_feature.reshape((-1, self.features_shape['map'][0], self.features_shape['map'][1])).unsqueeze(1)
        map_image = map_image.repeat(1, 3, 1, 1).float()  # Now shape [batch_size, 3, 90, 90]
        map_image = self.transform(map_image)  # Now shape [batch_size, 3, 224, 224]

        grasping_approach = grasping_approach.reshape((-1,self.features_shape['grasping_approach'][0], self.features_shape['grasping_approach'][1])).float()
        object_footprint = object_footprint.reshape((-1,self.features_shape['object_footprint'][0], self.features_shape['object_footprint'][1])).float()

        # Get the center vector
        x = x.long() # FIXME REMOVE BEFORE FLIGHT
        x_expanded = x.unsqueeze(-1).expand(-1, -1, self.features_shape['grasping_approach'][1])
        center = torch.gather(grasping_approach, 1, x_expanded).squeeze(1).float() # [batch, 1, 4] --> [batch, 4]

        center = self.embedder(center) # [batch, 4] --> [batch, 256]
        grasping_embedding = self.embedder(grasping_approach) # [batch, 150, 4] --> [batch, 150, 256]
        grasping_embedding_gru, _ = self.grasping_gru(grasping_embedding) # [batch, 150, 128] --> [batch, 150, 256]
        # grasping_embedding_gru = grasping_embedding_gru + grasping_embedding
        grasping_embedding_gru = F.layer_norm(grasping_embedding_gru + grasping_embedding, grasping_embedding_gru.size()[1:])
        grasping_embedding_gru = torch.mean(grasping_embedding_gru, dim=1) # [batch, 256]

        # process the footprint
        footprint_embedding = self.footprint_embedding(object_footprint) # [batch, 4, 2] --> [batch, 4, 32]
        footprint_embedding_gru, _ = self.footprint_gru(footprint_embedding) # [batch, 4, 16] --> [batch, 4, 32]
        footprint_embedding_gru = footprint_embedding_gru + footprint_embedding
        footprint_embedding_gru = torch.mean(footprint_embedding_gru, dim=1) # [batch, 32]

        # process the map image
        map_embedding = self.image_resnet(map_image) # [batch, 128]

        # Concatenate the embeddings
        # center: [batch, 256], grasping_embedding_gru: [batch, 256], map_embedding: [batch, 128], footprint_embedding_gru: [batch, 16]
        embedding = torch.cat([center, grasping_embedding_gru, map_embedding, footprint_embedding_gru], dim=1) # [batch, 400]

        return self.center_embedding(embedding)

class ContextEmbeddingResNetv3(nn.Module):
    def __init__(self, input_dim, embedding_dim, feature_structure, dropout=0.1):
        super(ContextEmbeddingResNetv3, self).__init__()
        self.feature_structure = feature_structure
        self.features_flatten_shape = self.feature_structure['features_structured']
        self.features_shape = self.feature_structure['features_shape']
        self.split_sizes = [self.features_flatten_shape[key] for key in ['map', 'object_footprint', 'grasping_approach']]

        # Vector Center embedding
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            # nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            # nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            # nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            # nn.BatchNorm1d(128)     # regularization
        )

        # Object footprint embedding
        self.footprint_embedding = nn.Sequential(
            nn.Linear(2, 8),
            # nn.LayerNorm(8),
            nn.ReLU(),
            nn.Linear(8, 16),
            # nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            # nn.BatchNorm1d(16)      # regularization
        )

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize(224),  # Resize to 224x224 antialias=True
            transforms.Grayscale(num_output_channels=3),  # Convert to three-channel grayscale
            transforms.Normalize(                           # ImageNet statistics
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

        # Map embedding ()
        # self.image_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_resnet = models.resnet18(pretrained=True)
        # Freeze the layers except the last one
        for name, param in self.image_resnet.named_parameters():
            # keep early layers frozen, fine-tune high-level features
            param.requires_grad = ("layer4" in name) or ("fc" in name)

        # Change the last layer
        self.image_resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            # nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # grasp embedding
        self.grasping_gru = nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)#, dropout=dropout)
        # object footprint embedding
        self.footprint_gru = nn.GRU(input_size=32, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)#, dropout=dropout)

        # context embedding
        self.context_embedding_gru = nn.GRU(input_size=256+256+128+32, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)#, dropout=dropout)
        self.context_embedding = nn.Sequential(
            nn.Linear(2*256, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            # nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x, feature):
        # Get the variables and convert to the correct format
        map_feature, object_footprint, grasping_approach = feature.split(self.split_sizes, dim=1)
        map_image = map_feature.reshape((-1, self.features_shape['map'][0], self.features_shape['map'][1])).unsqueeze(1)
        map_image = map_image.repeat(1, 3, 1, 1).float()  # Now shape [batch_size, 3, 90, 90]
        map_image = self.transform(map_image)  # Now shape [batch_size, 3, 224, 224]

        grasping_approach = grasping_approach.reshape((-1,self.features_shape['grasping_approach'][0], self.features_shape['grasping_approach'][1])).float()
        object_footprint = object_footprint.reshape((-1,self.features_shape['object_footprint'][0], self.features_shape['object_footprint'][1])).float()

        x = x.long() # FIXME REMOVE BEFORE FLIGHT
        x_expanded = x.unsqueeze(-1).expand(-1, -1, 4)
        # Get the context tensor - contex shape [batch, max_length, 4]
        context = torch.gather(grasping_approach, 1, x_expanded)
        context = self.embedder(context) # [batch, max_length, 4] --> [batch, max_length, 256]

        grasping_embedding = self.embedder(grasping_approach) # [batch, 150, 4] --> [batch, 150, 256]
        grasping_embedding_gru, _ = self.grasping_gru(grasping_embedding)
        # grasping_embedding_gru = grasping_embedding_gru + grasping_embedding
        grasping_embedding_gru = F.layer_norm(grasping_embedding_gru + grasping_embedding, grasping_embedding_gru.size()[1:])
        grasping_embedding_gru = torch.mean(grasping_embedding_gru, dim=1).unsqueeze(1).expand(-1, context.shape[1], -1) # [batch, 1, 256]

        # process the footprint
        footprint_embedding = self.footprint_embedding(object_footprint) # [batch, 4, 2] --> [batch, 4, 32]
        footprint_embedding_gru, _ = self.footprint_gru(footprint_embedding) # [batch, 4, 16] --> [batch, 4, 32]
        footprint_embedding_gru = footprint_embedding_gru + footprint_embedding
        footprint_embedding_gru = torch.mean(footprint_embedding_gru, dim=1).unsqueeze(1).expand(-1, context.shape[1], -1) # [batch, 1, 32]

        # process the map image
        map_embedding = self.image_resnet(map_image).unsqueeze(1).expand(-1, context.shape[1], -1) # [batch, 1, 128]

        embedding = torch.cat([context, grasping_embedding_gru, map_embedding, footprint_embedding_gru], dim=2) # [batch, max_length, 256*2 + 128 + 32]
        # embedding = context + grasping_embedding_gru
        embedding_context, _ = self.context_embedding_gru(embedding) # [batch, max_length, 256*2 + 128 + 32] --> [batch, max_length, 2*256]
        # embedding_context = embedding_context + embedding # Residual connection
        embedding_context = self.context_embedding(embedding_context)
        return embedding_context

###############################################################################################################
######################################## Model v4 with LayerNorm ##############################################
###############################################################################################################

class CenterEmbeddingResNetv4(nn.Module):
    def __init__(self, input_dim, embedding_dim, feature_structure, dropout=0.1):
        super(CenterEmbeddingResNetv4, self).__init__()
        self.feature_structure = feature_structure
        self.features_flatten_shape = self.feature_structure['features_structured']
        self.features_shape = self.feature_structure['features_shape']
        self.split_sizes = [self.features_flatten_shape[key] for key in ['map', 'object_footprint', 'grasping_approach']]

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize(224),  # Resize to 224x224 antialias=True
            transforms.Grayscale(num_output_channels=3),  # Convert to three-channel grayscale
            transforms.Normalize(                           # ImageNet statistics
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

        # Vector Center embedding
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        # Object footprint embedding
        self.footprint_embedding = nn.Sequential(
            nn.Linear(2, 8),
            nn.LayerNorm(8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            # nn.BatchNorm1d(16)      # regularization
        )

        # Map embedding ()
        # self.image_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_resnet = models.resnet18(pretrained=True)
        # Freeze the layers except the last one
        for name, param in self.image_resnet.named_parameters():
            # keep early layers frozen, fine-tune high-level features
            param.requires_grad = ("layer4" in name) or ("fc" in name)

        # Change the last layer
        self.image_resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # grasp embedding
        self.grasping_gru = nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)#, dropout=dropout)

        # object footprint embedding
        self.footprint_gru = nn.GRU(input_size=32, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)#, dropout=dropout)

        # center embedding
        self.center_embedding = nn.Sequential(
            nn.Linear(256+256+128+32, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),        # regularization
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            # nn.BatchNorm1d(128),
            nn.Dropout(dropout),        # regularization
            # nn.BatchNorm1d(128),    # regularization
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x, feature):
        # Get the variables and convert to the correct format
        map_feature, object_footprint, grasping_approach = feature.split(self.split_sizes, dim=1) # TODO: FINISH AND TEST
        map_image = map_feature.reshape((-1, self.features_shape['map'][0], self.features_shape['map'][1])).unsqueeze(1)
        map_image = map_image.repeat(1, 3, 1, 1).float()  # Now shape [batch_size, 3, 90, 90]
        map_image = self.transform(map_image)  # Now shape [batch_size, 3, 224, 224]

        grasping_approach = grasping_approach.reshape((-1,self.features_shape['grasping_approach'][0], self.features_shape['grasping_approach'][1])).float()
        object_footprint = object_footprint.reshape((-1,self.features_shape['object_footprint'][0], self.features_shape['object_footprint'][1])).float()

        # Get the center vector
        x = x.long() # FIXME REMOVE BEFORE FLIGHT
        x_expanded = x.unsqueeze(-1).expand(-1, -1, self.features_shape['grasping_approach'][1])
        center = torch.gather(grasping_approach, 1, x_expanded).squeeze(1).float() # [batch, 1, 4] --> [batch, 4]

        center = self.embedder(center) # [batch, 4] --> [batch, 256]
        grasping_embedding = self.embedder(grasping_approach) # [batch, 150, 4] --> [batch, 150, 256]
        grasping_embedding_gru, _ = self.grasping_gru(grasping_embedding) # [batch, 150, 128] --> [batch, 150, 256]
        # grasping_embedding_gru = grasping_embedding_gru + grasping_embedding
        grasping_embedding_gru = F.layer_norm(grasping_embedding_gru + grasping_embedding, grasping_embedding_gru.size()[1:])
        grasping_embedding_gru = torch.mean(grasping_embedding_gru, dim=1) # [batch, 256]

        # process the footprint
        footprint_embedding = self.footprint_embedding(object_footprint) # [batch, 4, 2] --> [batch, 4, 32]
        footprint_embedding_gru, _ = self.footprint_gru(footprint_embedding) # [batch, 4, 16] --> [batch, 4, 32]
        footprint_embedding_gru = footprint_embedding_gru + footprint_embedding
        footprint_embedding_gru = torch.mean(footprint_embedding_gru, dim=1) # [batch, 32]

        # process the map image
        map_embedding = self.image_resnet(map_image) # [batch, 128]

        # Concatenate the embeddings
        # center: [batch, 256], grasping_embedding_gru: [batch, 256], map_embedding: [batch, 128], footprint_embedding_gru: [batch, 16]
        embedding = torch.cat([center, grasping_embedding_gru, map_embedding, footprint_embedding_gru], dim=1) # [batch, 400]

        return self.center_embedding(embedding)

class ContextEmbeddingResNetv4(nn.Module):
    def __init__(self, input_dim, embedding_dim, feature_structure, dropout=0.1):
        super(ContextEmbeddingResNetv4, self).__init__()
        self.feature_structure = feature_structure
        self.features_flatten_shape = self.feature_structure['features_structured']
        self.features_shape = self.feature_structure['features_shape']
        self.split_sizes = [self.features_flatten_shape[key] for key in ['map', 'object_footprint', 'grasping_approach']]

        # Vector Center embedding
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            # nn.BatchNorm1d(128)     # regularization
        )

        # Object footprint embedding
        self.footprint_embedding = nn.Sequential(
            nn.Linear(2, 8),
            nn.LayerNorm(8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            # nn.BatchNorm1d(16)      # regularization
        )

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize(224),  # Resize to 224x224 antialias=True
            transforms.Grayscale(num_output_channels=3),  # Convert to three-channel grayscale
            transforms.Normalize(                           # ImageNet statistics
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

        # Map embedding ()
        # self.image_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_resnet = models.resnet18(pretrained=True)
        # Freeze the layers except the last one
        for name, param in self.image_resnet.named_parameters():
            # keep early layers frozen, fine-tune high-level features
            param.requires_grad = ("layer4" in name) or ("fc" in name)

        # Change the last layer
        self.image_resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # grasp embedding
        self.grasping_gru = nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)#, dropout=dropout)
        # object footprint embedding
        self.footprint_gru = nn.GRU(input_size=32, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)#, dropout=dropout)

        # context embedding
        self.context_embedding_gru = nn.GRU(input_size=256+256+128+32, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)#, dropout=dropout)
        self.context_embedding = nn.Sequential(
            nn.Linear(2*256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x, feature):
        # Get the variables and convert to the correct format
        map_feature, object_footprint, grasping_approach = feature.split(self.split_sizes, dim=1)
        map_image = map_feature.reshape((-1, self.features_shape['map'][0], self.features_shape['map'][1])).unsqueeze(1)
        map_image = map_image.repeat(1, 3, 1, 1).float()  # Now shape [batch_size, 3, 90, 90]
        map_image = self.transform(map_image)  # Now shape [batch_size, 3, 224, 224]

        grasping_approach = grasping_approach.reshape((-1,self.features_shape['grasping_approach'][0], self.features_shape['grasping_approach'][1])).float()
        object_footprint = object_footprint.reshape((-1,self.features_shape['object_footprint'][0], self.features_shape['object_footprint'][1])).float()

        x = x.long() # FIXME REMOVE BEFORE FLIGHT
        x_expanded = x.unsqueeze(-1).expand(-1, -1, 4)
        # Get the context tensor - contex shape [batch, max_length, 4]
        context = torch.gather(grasping_approach, 1, x_expanded)
        context = self.embedder(context) # [batch, max_length, 4] --> [batch, max_length, 256]

        grasping_embedding = self.embedder(grasping_approach) # [batch, 150, 4] --> [batch, 150, 256]
        grasping_embedding_gru, _ = self.grasping_gru(grasping_embedding)
        # grasping_embedding_gru = grasping_embedding_gru + grasping_embedding
        grasping_embedding_gru = F.layer_norm(grasping_embedding_gru + grasping_embedding, grasping_embedding_gru.size()[1:])
        grasping_embedding_gru = torch.mean(grasping_embedding_gru, dim=1).unsqueeze(1).expand(-1, context.shape[1], -1) # [batch, 1, 256]

        # process the footprint
        footprint_embedding = self.footprint_embedding(object_footprint) # [batch, 4, 2] --> [batch, 4, 32]
        footprint_embedding_gru, _ = self.footprint_gru(footprint_embedding) # [batch, 4, 16] --> [batch, 4, 32]
        footprint_embedding_gru = footprint_embedding_gru + footprint_embedding
        footprint_embedding_gru = torch.mean(footprint_embedding_gru, dim=1).unsqueeze(1).expand(-1, context.shape[1], -1) # [batch, 1, 32]

        # process the map image
        map_embedding = self.image_resnet(map_image).unsqueeze(1).expand(-1, context.shape[1], -1) # [batch, 1, 128]

        embedding = torch.cat([context, grasping_embedding_gru, map_embedding, footprint_embedding_gru], dim=2) # [batch, max_length, 256*2 + 128 + 32]
        # embedding = context + grasping_embedding_gru
        embedding_context, _ = self.context_embedding_gru(embedding) # [batch, max_length, 256*2 + 128 + 32] --> [batch, max_length, 2*256]
        # embedding_context = embedding_context + embedding # Residual connection
        embedding_context = self.context_embedding(embedding_context)
        return embedding_context

###############################################################################################################
##################################### Model v4 without footprint ##############################################
###############################################################################################################

# ORIGINAL MAX PERFORMANCE
class CenterEmbeddingResNetv5(nn.Module):
    def __init__(self, input_dim, embedding_dim, feature_structure, dropout=0.1):
        super(CenterEmbeddingResNetv5, self).__init__()
        self.feature_structure = feature_structure
        self.features_flatten_shape = self.feature_structure['features_structured']
        self.features_shape = self.feature_structure['features_shape']
        self.split_sizes = [self.features_flatten_shape[key] for key in ['map', 'object_footprint', 'grasping_approach']]

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize(224),  # Resize to 224x224 antialias=True
            transforms.Grayscale(num_output_channels=3),  # Convert to three-channel grayscale
            transforms.Normalize(                           # ImageNet statistics
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

        # Vector Center embedding
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            # nn.BatchNorm1d(128)     # regularization
        )

        # Object footprint embedding
        self.footprint_embedding = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            # nn.BatchNorm1d(16)      # regularization
        )

        # Map embedding ()
        # self.image_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_resnet = models.resnet18(pretrained=True)
        # Freeze the layers except the last one
        for name, param in self.image_resnet.named_parameters():
            # keep early layers frozen, fine-tune high-level features
            param.requires_grad = ("layer4" in name) or ("fc" in name)

        # Change the last layer
        self.image_resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # grasp embedding
        self.grasping_gru = nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)#, dropout=dropout)

        # object footprint embedding
        self.footprint_gru = nn.GRU(input_size=32, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)#, dropout=dropout)

        # center embedding
        self.center_embedding = nn.Sequential(
            nn.Linear(256+256+128+32, 256),
            nn.ReLU(),
            nn.Dropout(dropout),        # regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),        # regularization
            # nn.BatchNorm1d(128),    # regularization
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x, feature):
        # Get the variables and convert to the correct format
        map_feature, object_footprint, grasping_approach = feature.split(self.split_sizes, dim=1) # TODO: FINISH AND TEST
        map_image = map_feature.reshape((-1, self.features_shape['map'][0], self.features_shape['map'][1])).unsqueeze(1)
        map_image = map_image.repeat(1, 3, 1, 1).float()  # Now shape [batch_size, 3, 90, 90]
        map_image = self.transform(map_image)  # Now shape [batch_size, 3, 224, 224]

        grasping_approach = grasping_approach.reshape((-1,self.features_shape['grasping_approach'][0], self.features_shape['grasping_approach'][1])).float()
        object_footprint = object_footprint.reshape((-1,self.features_shape['object_footprint'][0], self.features_shape['object_footprint'][1])).float()

        # Get the center vector
        x = x.long() # FIXME REMOVE BEFORE FLIGHT
        x_expanded = x.unsqueeze(-1).expand(-1, -1, self.features_shape['grasping_approach'][1])
        center = torch.gather(grasping_approach, 1, x_expanded).squeeze(1).float() # [batch, 1, 4] --> [batch, 4]

        center = self.embedder(center) # [batch, 4] --> [batch, 256]
        grasping_embedding = self.embedder(grasping_approach) # [batch, 150, 4] --> [batch, 150, 256]
        grasping_embedding_gru, _ = self.grasping_gru(grasping_embedding) # [batch, 150, 128] --> [batch, 150, 256]
        # grasping_embedding_gru = grasping_embedding_gru + grasping_embedding
        grasping_embedding_gru = F.layer_norm(grasping_embedding_gru + grasping_embedding, grasping_embedding_gru.size()[1:])
        grasping_embedding_gru = torch.mean(grasping_embedding_gru, dim=1) # [batch, 256]

        # process the footprint
        footprint_embedding = self.footprint_embedding(object_footprint) # [batch, 4, 2] --> [batch, 4, 32]
        footprint_embedding_gru, _ = self.footprint_gru(footprint_embedding) # [batch, 4, 16] --> [batch, 4, 32]
        footprint_embedding_gru = footprint_embedding_gru + footprint_embedding
        footprint_embedding_gru = torch.mean(footprint_embedding_gru, dim=1) # [batch, 32]

        # process the map image
        map_embedding = self.image_resnet(map_image) # [batch, 128]

        # Concatenate the embeddings
        # center: [batch, 256], grasping_embedding_gru: [batch, 256], map_embedding: [batch, 128], footprint_embedding_gru: [batch, 16]
        embedding = torch.cat([center, grasping_embedding_gru, map_embedding, footprint_embedding_gru], dim=1) # [batch, 400]

        return self.center_embedding(embedding)

class ContextEmbeddingResNetv5(nn.Module):
    def __init__(self, input_dim, embedding_dim, feature_structure, dropout=0.1):
        super(ContextEmbeddingResNetv5, self).__init__()
        self.feature_structure = feature_structure
        self.features_flatten_shape = self.feature_structure['features_structured']
        self.features_shape = self.feature_structure['features_shape']
        self.split_sizes = [self.features_flatten_shape[key] for key in ['map', 'object_footprint', 'grasping_approach']]

        # Vector Center embedding
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            # nn.BatchNorm1d(128)     # regularization
        )

        # Object footprint embedding
        self.footprint_embedding = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            # nn.BatchNorm1d(16)      # regularization
        )

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize(224),  # Resize to 224x224 antialias=True
            transforms.Grayscale(num_output_channels=3),  # Convert to three-channel grayscale
            transforms.Normalize(                           # ImageNet statistics
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

        # Map embedding ()
        # self.image_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_resnet = models.resnet18(pretrained=True)
        # Freeze the layers except the last one
        for name, param in self.image_resnet.named_parameters():
            # keep early layers frozen, fine-tune high-level features
            param.requires_grad = ("layer4" in name) or ("fc" in name)

        # Change the last layer
        self.image_resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # grasp embedding
        self.grasping_gru = nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)#, dropout=dropout)
        # object footprint embedding
        self.footprint_gru = nn.GRU(input_size=32, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)#, dropout=dropout)

        # context embedding
        self.context_embedding_gru = nn.GRU(input_size=256+256+128+32, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)#, dropout=dropout)
        self.context_embedding = nn.Sequential(
            nn.Linear(2*256, 256),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.BatchNorm1d(128),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x, feature):
        # Get the variables and convert to the correct format
        map_feature, object_footprint, grasping_approach = feature.split(self.split_sizes, dim=1)
        map_image = map_feature.reshape((-1, self.features_shape['map'][0], self.features_shape['map'][1])).unsqueeze(1)
        map_image = map_image.repeat(1, 3, 1, 1).float()  # Now shape [batch_size, 3, 90, 90]
        map_image = self.transform(map_image)  # Now shape [batch_size, 3, 224, 224]

        grasping_approach = grasping_approach.reshape((-1,self.features_shape['grasping_approach'][0], self.features_shape['grasping_approach'][1])).float()
        object_footprint = object_footprint.reshape((-1,self.features_shape['object_footprint'][0], self.features_shape['object_footprint'][1])).float()

        x = x.long() # FIXME REMOVE BEFORE FLIGHT
        x_expanded = x.unsqueeze(-1).expand(-1, -1, 4)
        # Get the context tensor - contex shape [batch, max_length, 4]
        context = torch.gather(grasping_approach, 1, x_expanded)
        context = self.embedder(context) # [batch, max_length, 4] --> [batch, max_length, 256]

        grasping_embedding = self.embedder(grasping_approach) # [batch, 150, 4] --> [batch, 150, 256]
        grasping_embedding_gru, _ = self.grasping_gru(grasping_embedding)
        # grasping_embedding_gru = grasping_embedding_gru + grasping_embedding
        grasping_embedding_gru = F.layer_norm(grasping_embedding_gru + grasping_embedding, grasping_embedding_gru.size()[1:])
        grasping_embedding_gru = torch.mean(grasping_embedding_gru, dim=1).unsqueeze(1).expand(-1, context.shape[1], -1) # [batch, 1, 256]

        # process the footprint
        footprint_embedding = self.footprint_embedding(object_footprint) # [batch, 4, 2] --> [batch, 4, 32]
        footprint_embedding_gru, _ = self.footprint_gru(footprint_embedding) # [batch, 4, 16] --> [batch, 4, 32]
        footprint_embedding_gru = footprint_embedding_gru + footprint_embedding
        footprint_embedding_gru = torch.mean(footprint_embedding_gru, dim=1).unsqueeze(1).expand(-1, context.shape[1], -1) # [batch, 1, 32]

        # process the map image
        map_embedding = self.image_resnet(map_image).unsqueeze(1).expand(-1, context.shape[1], -1) # [batch, 1, 128]

        embedding = torch.cat([context, grasping_embedding_gru, map_embedding, footprint_embedding_gru], dim=2) # [batch, max_length, 256*2 + 128 + 32]
        # embedding = context + grasping_embedding_gru
        embedding_context, _ = self.context_embedding_gru(embedding) # [batch, max_length, 256*2 + 128 + 32] --> [batch, max_length, 2*256]
        # embedding_context = embedding_context + embedding # Residual connection
        embedding_context = self.context_embedding(embedding_context)
        return embedding_context
