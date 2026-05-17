# Reorganized from train/TrainNS_dataloader.py
# Original author:
# Description: Dataset loading, negative sampling, and batching utilities for cooperative grasping data

import os
import torch
import pickle
import random
import collections
import numpy as np
from sklearn.metrics import pairwise_distances
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset as TorchDataset
from conditional_embedding_model.utils import Dataset

def divide_dataset(dataset, test_prop=0.1, val_prop=0.2):
    dataset_length = len(dataset)

    # Calculate the absolute sizes of each split
    test_size = int(dataset_length * test_prop)
    val_size = int(dataset_length * val_prop)
    train_size = dataset_length - test_size - val_size  # Ensure all samples are used

    print(f"Dataset length: {dataset_length}")
    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

    # Perform the split
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, validation_dataset, test_dataset


# Function to define the center and context and situation
def create_center_context_lists(dataset):
    """
    Optimized function to output a list of centers and a corresponding list of lists of contexts without using dictionaries.

    Args:
    - dataset: An instance of the Dataset class with the 'combinations' attribute filled.

    Returns:
    - centers_list: A sorted list of unique centers.
    - contexts_list: A list of lists, where each sublist contains contexts for the corresponding center in centers_list.
    - features_list: A list of dictionaries, where each dictionary contains the features of the corresponding center in centers_list.
    """
    # create a new combination list with the two first rows interchanged.
    combinations_reversed = np.copy(dataset.combinations)
    combinations_reversed[:, 0] = dataset.combinations[:, 1]
    combinations_reversed[:, 1] = dataset.combinations[:, 0]

    combinations = np.concatenate((dataset.combinations, combinations_reversed), axis=0)

    # Filter combinations where the third column equals one
    valid_combinations = combinations[combinations[:, 2] == 1]

    # Initialize lists to hold centers and their contexts
    centers = valid_combinations[:, 0]
    contexts = valid_combinations[:, 1]

    # Find unique centers for the list
    unique_centers = np.unique(centers)

    # Initialize the contexts list with empty lists for each center
    contexts_list = [[] for _ in range(len(unique_centers))]

    # Populate the contexts list
    for center, context in zip(centers, contexts):
        index = np.where(unique_centers == center)[0][0]  # Find the index of the center in unique_centers
        contexts_list[index].append(context)

    # Convert unique_centers to a list for consistency with the expected return type
    centers_list = unique_centers.tolist()

    # flatten and build a np tensor of the features
    map_feature = dataset.map.flatten()
    object_footprint_feature = dataset.object_footprint.flatten()
    grasping_approach_feature = dataset.grasping_approach.flatten()
    features = np.concatenate((map_feature, object_footprint_feature, grasping_approach_feature), axis=0)

    # create dict with the flatten features shapes and the original shapes
    features_structured = {'map': map_feature.shape[0], 'object_footprint': object_footprint_feature.shape[0], 'grasping_approach': grasping_approach_feature.shape[0]}
    features_shape = {'map': dataset.map.shape, 'object_footprint': dataset.object_footprint.shape, 'grasping_approach': dataset.grasping_approach.shape}

    features_struct_dict = {'features_structured': features_structured, 'features_shape': features_shape}
    features_list = [features for _ in range(len(centers_list))]

    return centers_list, contexts_list, features_list, features_struct_dict

def extract_dataset(general_dataset):
    """
    Extracts the centers, contexts, and features from a dataset.

    Args:
    - general_dataset: A list of instances of the Dataset class.

    Returns:
    - centers: A list of centers.
    - contexts: A list of lists, where each sublist contains contexts for the corresponding center in centers.
    - features: A list of dictionaries, where each dictionary contains the features of the corresponding center in centers.
    """
    centers, contexts, features = [], [], []
    bad_quality = 0
    for situation in general_dataset:
        c, co, f, fst = create_center_context_lists(situation)
        if len(c) > 0:
            centers += c
            contexts += co
            features += f
        else:
            bad_quality += 1
    return centers, contexts, features, bad_quality, fst

#@save
def get_negatives(contexts, centers, total_length=None, K=1.0):
    """Return noise words in negative sampling.
    K = 1.0 means one negative sample for each context word.
    If total_length is specified, it will be used as the total number of negative samples for each context.
    If total_length is None, it will be set to K * maximum context length
    """
    # If total_length not specified, use K * maximum context length as default
    if total_length is None:
        max_context_len = max(len(context) for context in contexts)
        total_length = int(max_context_len * (K))

    # Calculate sampling weights
    unique_centers = list(set(centers))
    counter = collections.Counter(centers)
    unique_sampling_weights = {center: counter[center]**0.75 for center in unique_centers}
    sampling_weights = list(unique_sampling_weights.values())

    all_negatives = []
    generator = RandomGenerator(sampling_weights)

    for context in contexts:
        # calculate the needed negative to reach the total_length
        neg_count = total_length - len(context)
        neg_count = max(neg_count, 0)  # Ensure non-negative count

        # generate negatives for this context
        negatives = []
        while len(negatives) < neg_count:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in context:
                negatives.append(neg)
        all_negatives.append(negatives)

    return all_negatives


def get_nearest_negatives(center_features, candidate_features, contexts, total_length=None, K=1.0):
    """
    Sample nearest negatives based on original feature-space distance.

    Args:
        center_features (np.ndarray): [num_centers, feature_dim] feature vectors of centers.
        candidate_features (np.ndarray): [num_candidates, feature_dim] all possible candidate feature vectors.
        contexts (list[list[int]]): context indices for each center.
        total_length (int, optional): total context + negative length per center.
        K (float): multiplier to determine total length if not explicitly set.

    Returns:
        list[list[int]]: negatives indices for each center.
    """

    num_centers = len(center_features)

    # Compute distance matrix [num_centers, num_candidates]
    distances = pairwise_distances(center_features, candidate_features, metric='euclidean')

    all_negatives = []

    for idx in range(num_centers):
        context_indices = set(contexts[idx])

        num_context = len(context_indices)
        neg_count = total_length - num_context if total_length else int(num_context * K)
        neg_count = max(neg_count, 0)

        # Sort candidates by distance (ascending)
        nearest_candidates = np.argsort(distances[idx])

        negatives = []
        for cand_idx in nearest_candidates:
            if cand_idx not in context_indices:
                negatives.append(cand_idx)
            if len(negatives) >= neg_count:
                break

        all_negatives.append(negatives)

    return all_negatives

#@save
def get_negatives_fixed_neg_length(contexts, centers, K=1.0):
    """Return noise words in negative sampling."""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the excluded unknown node)
    unique_centers = list(set(centers))  # Get unique centers
    counter = collections.Counter(centers)  # Count the frequency of each center
    # Calculate sampling weights for unique centers
    unique_sampling_weights = {center: counter[center]**0.75 for center in unique_centers}
    sampling_weights = list(unique_sampling_weights.values())
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    max_len = max(len(context) for context in contexts)
    for context in contexts:
        negatives = []
        while len(negatives) < max_len * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in context:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

# Define the torch dataset class
class GraspDataset(TorchDataset):
    def __init__(self, centers, contexts, features, negatives):
            assert len(centers) == len(contexts) == len(negatives) == len(features)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives
            self.features = features

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index], self.features[index])

# batchify function
def batchify_wrapper(max_length):
    def batchify(data):
        """Return a minibatch of examples for skip-gram with negative sampling."""
        # max_len = max(len(c) + len(n) for _, c, n, _ in data)
        max_len = max_length
        centers, contexts_negatives, masks, labels, features = [], [], [], [], []
        for center, context, negative, feature in data:
            combined = context + negative
            combined_labels = [1] * len(context) + [0] * len(negative)

            # Zip, shuffle, and unzip
            combined_pairs = list(zip(combined, combined_labels))
            random.shuffle(combined_pairs)
            shuffled_combined, shuffled_labels = zip(*combined_pairs)

            cur_len = len(shuffled_combined)
            centers += [center]
            contexts_negatives += [list(shuffled_combined) + [0] * (max_len - cur_len)]
            masks += [[1] * cur_len + [0] * (max_len - cur_len)]
            labels += [list(shuffled_labels) + [0] * (max_len - cur_len)]
            features += [feature]
        return (torch.tensor(centers).reshape(-1, 1), torch.tensor(contexts_negatives),
                torch.tensor(masks), torch.tensor(labels), torch.tensor(features))
    return batchify

#@save
class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights):
        # Exclude
        # self.sampling_weights = sampling_weights
        self.sampling_weights = np.ones_like(sampling_weights)
        self.population = list(range(1, len(sampling_weights) + 1))
        self.canditates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.canditates):
            # Cache 'k' random sampling results
            self.canditates = random.choices(self.population, self.sampling_weights, k=10000)
            # self.candidates = random.sample(self.population, len(self.population))
            self.i = 0
        self.i += 1
        return self.canditates[self.i - 1]

#@save
def load_data_grasp(batch_size, multiplier_factor, dataset_path, max_length=100):
    """Download the CoopGrasping-v0 dataset and return data iterators
    Args:
        batch_size (int): The batch size for the data iterator.
        multiplier_factor (int): The factor to multiply the number of negatives for each context.
        dataset_path (str): The path to the dataset file.
        max_length (int, optional): The maximum length of the contexts and negatives. Defaults to 100.

    Returns:
        data_iter (DataLoader): The data iterator for the dataset.
        dataset (GraspDataset): The dataset containing the centers, contexts, features, and negatives.
        feature_structure (list): The structure of the features in the dataset.

    """
    with open(dataset_path, 'rb') as file:
        general_dataset = list((pickle.load(file)).values())
    # Extract the centers, contexts, and features from the dataset
    centers, contexts, features, _, feature_structure = extract_dataset(general_dataset)
    # Compute negatives for all contexts
    # all_negatives = get_nearest_negatives(contexts, centers, K=multiplier_factor)
    all_negatives = get_negatives(contexts, centers, K=multiplier_factor)
    # Create the dataset and data iterator
    dataset = GraspDataset(centers, contexts, features, all_negatives) # Can be computed max_lenght here
    train_dataset, validation_dataset, test_dataset = divide_dataset(dataset)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=batchify_wrapper(max_length))
    validation_dataloader = DataLoader(validation_dataset, batch_size, shuffle=True, collate_fn=batchify_wrapper(max_length))
    return train_dataloader, validation_dataloader, test_dataset, feature_structure

if __name__ == "__main__":
    batch_size = 64
    max_length = 60
    multiplier_factor = 1.2
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, '../config/data/CoopGrasping-v6_dataset.pkl')
    train_dataloader, validation_dataloader, test_dataset, feature_structure = load_data_grasp(batch_size, multiplier_factor, dataset_path, max_length)
    for batch in train_dataloader:
        for name, data in zip(['centers', 'contexts_negatives', 'masks', 'labels', 'features'], batch):
            print(name, 'shape:', data.shape) if name != 'features' else print(name, 'length:', len(data))
        break
