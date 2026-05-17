# Reorganized from train/TrainNS_Eval.py
# Original author:
# Description: Post-training evaluation routines including top-k accuracy, ROC/PR curves, t-SNE visualizations, and statistical tests

import os
import time
import yaml
import torch
import wandb
import pickle
import itertools
import numpy as np
from torch import nn
import seaborn as sns
from scipy import stats
# from simulation.utils import *
# from simulation.utils import *
from TrainNS_train import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from TrainNS_dataloader import *
from scipy.stats import binomtest
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model(net, test_dataloader, device='cuda:0', verbose=True, cosine=True, method='wieghted'):
    # device = 'cuda:0'
    all_preds = []
    all_labels = []
    all_masks = []
    net.eval()
    for center, context_negatives, masks, labels, feature in test_dataloader:
        center, context_negatives, masks, labels, feature = center.to(device), context_negatives.to(device), masks.to(device), labels.to(device), feature.to(device)
        cos_similarities = get_similar_vectors(center, context_negatives, net[0], net[1], feature, cosine)
        # Multiply by mask to get the correct similarities
        prediction = cos_similarities * masks
        # evaluate_metrics(labels.cpu().detach().numpy(), prediction.cpu().detach().numpy())
        all_preds.append(prediction)
        all_labels.append(labels)
        all_masks.append(masks)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Flatten
    pred = all_preds.view(-1)
    labels = all_labels.view(-1)
    masks = all_masks.view(-1)

    # Filter for valid positions only
    valid_indices = (masks == 1)
    pred_valid = pred[valid_indices]
    labels_valid = labels[valid_indices]

    # Get the predictions and labels
    labels = labels_valid.cpu().detach().numpy()
    pred = pred_valid.cpu().detach().numpy()

    threshold_pos = optimize_threshold(labels, pred)
    print(f'total samples: {labels.size}')
    evaluate_metrics_optimized(labels, pred, threshold_pos, verbose, method)
    roc_auc = plot_roc_curve(labels, pred)
    plot_precision_recall_curve(labels, pred)
    plot_densities(labels, pred)

    # statistic analyzis
    # statistical_test(pred, labels, threshold_pos, roc_auc)
        # Add these new checks at the end
    # print("\n=== Feature Differences Analysis ===")
    # check_feature_differences(labels, pred)

    # print("\n=== Random Guessing Check ===")
    # check_random_guess(labels, pred)

    # # Evaluate vs. random guessing
    # evaluate_against_random_guessing(labels, pred, threshold_pos=threshold_pos)

    # Evaluate if a feature dimension differs between classes
    # evaluate_feature_difference(net, test_dataloader, feature_idx=0)

def evaluate_metrics(labels, pred, threshold_pos=0.6, verbose=True):
    # Binarize predictions based on the threshold

    normalized_pred = (pred + 1) / 2
    binarized_pred = (normalized_pred > threshold_pos).astype(int)

    # Calculate metrics
    accuracy = (labels == binarized_pred).mean()
    precision = precision_score(labels.flatten(), binarized_pred.flatten(), average='weighted')
    recall = recall_score(labels.flatten(), binarized_pred.flatten(), average='weighted')
    f1 = f1_score(labels.flatten(), binarized_pred.flatten(), average='weighted') # macro
    auc_roc = roc_auc_score(labels.flatten(), normalized_pred.flatten())

    if verbose:
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC-ROC: {auc_roc:.4f}')
    return accuracy, precision, recall, f1, auc_roc

def optimize_threshold(labels, pred, thresholds=np.arange(0.1, 1.0, 0.01)):
    best_f1 = 0
    best_threshold = 0.5
    for threshold in thresholds:
        _, _, _, f1, _ = evaluate_metrics(labels, pred, threshold, verbose=False)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    print(f'Optimal Threshold: {best_threshold}, Best F1 Score: {best_f1:.4f}')
    return best_threshold

def evaluate_model_topk(net, test_dataloader, device='cuda:0', verbose=True, cosine=True):
    # Evaluate the model using the top-k accuracy taking all the context vectors into account
    net.eval()
    # device = 'cuda:0'
    all_preds = []
    all_labels = []
    top_k_correct = {1: 0, 3: 0, 5: 0, 7: 0}
    total_samples = 0

    with torch.no_grad():
        for center, context_negatives, masks, labels, feature in test_dataloader:
            center, context_negatives, masks, labels, feature = center.to(device), context_negatives.to(device), masks.to(device), labels.to(device), feature.to(device)
            cos_similarities = get_similar_vectors(center, context_negatives, net[0], net[1], feature, cosine)

            # Multiply by mask to get the correct similarities
            prediction = cos_similarities.clone()
            prediction[masks == 0] = -float('inf')  # Set masked positions to -inf to ignore them in top-k
            all_preds.append(prediction)
            all_labels.append(labels)

            # Evaluate top-k metrics
            for k in top_k_correct.keys():
                _, top_k_indices = prediction.topk(k, dim=1)
                for i in range(labels.size(0)):  # iterate over the batch
                    true_label_indices = labels[i].nonzero(as_tuple=True)[0].tolist()
                    top_k_indices_batch = top_k_indices[i].tolist()
                    if any(idx in top_k_indices_batch for idx in true_label_indices):
                        top_k_correct[k] += 1

            # total_samples += labels.size(0) # original
            total_samples += (labels.sum(dim=1) > 0).sum().item()

    # Calculate top-k accuracy
    for k in top_k_correct.keys():
        top_k_accuracy = top_k_correct[k] / total_samples
        print(f'Top-{k} Accuracy: {top_k_accuracy:.4f}')

    return top_k_correct

def evaluate_model_topk_optimized(net, test_dataloader, device='cuda:0',
                        verbose=True, cosine=True):
    """
    Hit@k (top-k accuracy) evaluation.
    Rows that do **not** contain any positive label are excluded
    from the denominator, in line with standard retrieval metrics.
    """
    net.eval()
    top_k_correct   = {1: 0, 3: 0, 5: 0, 7: 0}
    rows_with_pos   = {1: 0, 3: 0, 5: 0, 7: 0}

    with torch.no_grad():
        for center, context_negatives, masks, labels, feature in test_dataloader:
            center, context_negatives, masks, labels, feature = (
                center.to(device), context_negatives.to(device),
                masks.to(device),  labels.to(device), feature.to(device)
            )

            cos_similarities = get_similar_vectors(center,
                                                   context_negatives,
                                                   net[0], net[1],
                                                   feature, cosine)

            # Mask out invalid positions so they can never be selected
            prediction = cos_similarities.clone()
            prediction[masks == 0] = -float('inf')

            # Boolean mask: which rows contain ≥1 positive?
            has_positive = (labels.sum(dim=1) > 0)

            for k in top_k_correct.keys():
                _, topk_idx = prediction.topk(k, dim=1)

                for i in range(labels.size(0)):          # iterate rows
                    if not has_positive[i]:
                        continue                         # skip negative-only rows

                    true_idx = labels[i].nonzero(as_tuple=True)[0]
                    hit = torch.isin(topk_idx[i], true_idx).any()
                    if hit:
                        top_k_correct[k] += 1

                # update denominator
                rows_with_pos[k] += has_positive.sum().item()

    # Print final hit@k
    for k in top_k_correct:
        denom = max(rows_with_pos[k], 1)                # avoid /0
        hit_at_k = top_k_correct[k] / denom
        print(f'Top-{k} Accuracy (hit@{k}): {hit_at_k:.4f}')

    return top_k_correct

def evaluate_metrics_optimized(labels, pred, threshold_pos=0.6, verbose=True, method='weighted'):
    """
    Compute accuracy, precision, recall, F1 and ROC-AUC.
    `pred` is the raw cosine-similarity in [-1,1]; we
    normalise to [0,1] exactly once and use that everywhere.
    """
    # Normalise to [0,1] for probabilistic metrics
    normalized_pred = (pred + 1) / 2

    # Binarise via threshold on the normalised score
    binarized_pred = (normalized_pred > threshold_pos).astype(int)

    # Metrics
    accuracy  = (labels == binarized_pred).mean()
    precision = precision_score(labels.flatten(),
                                binarized_pred.flatten(),
                                average=method,
                                zero_division=0)
    recall    = recall_score(labels.flatten(),
                             binarized_pred.flatten(),
                             average=method,
                             zero_division=0)
    f1        = f1_score(labels.flatten(),
                         binarized_pred.flatten(),
                         average=method,
                         zero_division=0)
    auc_roc   = roc_auc_score(labels.flatten(),
                              normalized_pred.flatten())

    if verbose:
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
              f'Recall: {recall:.4f}, F1: {f1:.4f}, AUC-ROC: {auc_roc:.4f}')
    return accuracy, precision, recall, f1, auc_roc

########################################################################################
# Statistical Analyzis functions
########################################################################################

def statistical_test(pred, labels, threshold_pos, auc_roc):
        # 1. P-value of the two distributions (label 0 and 1)
    positive_preds = pred[labels == 1]
    negative_preds = pred[labels == 0]

    # Handle cases where one group might be empty
    if len(positive_preds) == 0 or len(negative_preds) == 0:
        print("Warning: One of the classes has no samples. P-value test skipped.")
    else:
        # Perform one-tailed Mann-Whitney U test (positive_preds > negative_preds)
        try:
            stat, p_value_dist = mannwhitneyu(positive_preds, negative_preds, alternative='greater')
            print(f"\nP-value of distributions (Mann-Whitney U test): {p_value_dist:.4e}")
            if p_value_dist < 0.05:
                print("The two distributions are significantly different (p < 0.05)")
            else:
                print("The two distributions are not significantly different")
        except ValueError as e:
            print(f"Error in Mann-Whitney U test: {e}")

    # 2. Check if classifier beats random guessing
    # a. For accuracy: binomial test
    normalized_pred = (pred + 1) / 2
    binarized_pred = (normalized_pred > threshold_pos).astype(int)
    n_correct = np.sum(binarized_pred == labels)
    n_total = labels.size

    if n_total == 0:
        print("Warning: No samples available for binomial test.")
    else:
        # Perform binomial test
        try:
            result = binomtest(n_correct, n_total, p=0.5, alternative='greater')
            p_value_acc = result.pvalue
            print(f"\nBinomial test for accuracy vs random guessing:")
            print(f"Correct predictions: {n_correct}/{n_total} = {n_correct/n_total:.4f}")
            print(f"P-value: {p_value_acc:.4e}")
            if p_value_acc < 0.05:
                print("Result: Accuracy is significantly better than random (p < 0.05)")
            else:
                print("Result: Accuracy is not significantly better than random")
        except Exception as e:
            print(f"Error in binomial test: {e}")

    # b. For AUC: use the p-value from Mann-Whitney test (if available)
    try:
        print(f"\nAUC: {auc_roc:.4f}")
        if 'p_value_dist' in locals():
            print(f"P-value from Mann-Whitney U test (AUC > 0.5): {p_value_dist:.4e}")
            if p_value_dist < 0.05:
                print("Result: AUC is significantly better than 0.5 (p < 0.05)")
            else:
                print("Result: AUC is not significantly better than 0.5")
    except NameError:
        print("AUC value not available for comparison.")

########################################################################################
# Plotting functions
########################################################################################

def plot_roc_curve(labels, pred, size=[8,6], title_font_size=20, font_size=16):
    normalized_pred = (pred.flatten() + 1) / 2
    fpr, tpr, thresholds = roc_curve(labels.flatten(), normalized_pred)

    # Calculate the Youden's J statistic
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]

    # Calculate the AUC
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=size)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    # Highlight the optimal cutpoint on the ROC curve
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label='Cutpoint (threshold = %0.2f)' % optimal_threshold)

    # Annotate the cutpoint
    plt.annotate('Cutpoint\nThreshold = %0.2f' % optimal_threshold,
                 xy=(fpr[optimal_idx], tpr[optimal_idx]),
                 xytext=(fpr[optimal_idx] + 0.1, tpr[optimal_idx] - 0.1),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=font_size-2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=font_size)
    plt.ylabel('True Positive Rate', fontsize=font_size)
    plt.title('Receiver Operating Characteristic', fontsize=title_font_size)
    plt.legend(loc="lower right", fontsize=font_size)
    plt.show()

    print(f'Optimal Threshold: {optimal_threshold}')
    print(f'FPR at Optimal Threshold: {fpr[optimal_idx]}')
    print(f'TPR at Optimal Threshold: {tpr[optimal_idx]}')
    return roc_auc

def plot_precision_recall_curve(labels, pred):
    normalized_pred = (pred.flatten() + 1) / 2
    precision, recall, thresholds = precision_recall_curve(labels.flatten(), normalized_pred)

    # Calculate the AUC for the Precision-Recall curve
    pr_auc = auc(recall, precision)

    # Plot the Precision-Recall curve
    plt.figure(figsize=(12, 12))
    plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.title('Precision-Recall Curve', fontsize=20)
    plt.legend(loc="lower left", fontsize=14)
    plt.show()

    print(f'Precision-Recall AUC: {pr_auc}')

def plot_densities(labels, pred, threshold=0.5):
    # Separate predictions into positive and negative cases based on labels
    # normalized_pred = (pred + 1) / 2
    positive_preds = pred[labels == 1]
    negative_preds = pred[labels == 0]

    print(f'Positive Predictions: {positive_preds.size}')
    print(f'Negative Predictions: {negative_preds.size}')

    # Create the density plot
    plt.figure(figsize=(10, 6))

    # Plot the density of positive predictions
    sns.kdeplot(positive_preds, shade=True, color="blue", label="Positive Cases")

    # Plot the density of negative predictions
    sns.kdeplot(negative_preds, shade=True, color="red", label="Negative Cases")

    # Add title and labels
    plt.title("Density Plot of Predictions")
    plt.xlabel("Prediction Value")
    plt.ylabel("Density")
    plt.legend()

    # Show plot
    plt.show()

def graph_2dspace(net, test_dataloader, idx=0, seed=42):
    device = 'cuda:0'
    np.random.seed(seed)
    for center, context_negatives, masks, labels, feature in test_dataloader:
        center, context_negatives, masks, labels, feature = center.to(device), context_negatives.to(device), masks.to(device), labels.to(device), feature.to(device)
        embed_v = net[0](center, feature).cpu().detach().numpy()  # Add batch dimension to query_vector if needed

        # Generate embeddings for context vectors using net_u
        embed_u = net[1](context_negatives, feature).cpu().detach().numpy()  # Assume contexts_vectors is batched if necessary
        # Stack the embeddings to run t-SNE on all of them together
        all_embeddings = np.vstack([embed_v[idx] , embed_u[idx]])

        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=seed)
        reduced_embeddings = tsne.fit_transform(all_embeddings)

        # Separate the reduced embeddings
        reduced_v = reduced_embeddings[0]
        reduced_u = reduced_embeddings[1:]

        # Plot
        plt.figure(figsize=(12, 12))
        plt.scatter(reduced_u[:, 0], reduced_u[:, 1], c='blue', label='Context Vectors u')
        plt.scatter(reduced_v[0], reduced_v[1], c='red', label='Center Vector v')

        # Optionally, draw lines from v to the closest vectors in u
        distances = np.sqrt(np.sum((reduced_u - reduced_v)**2, axis=1))
        closest_indices = np.argsort(distances)[:5]  # Get the indices of the 5 closest vectors
        for index in closest_indices:
            plt.plot([reduced_v[0], reduced_u[index, 0]], [reduced_v[1], reduced_u[index, 1]], 'grey', linestyle='--', alpha=0.5)

        plt.legend()
        # add the axis labels
        plt.xlabel('First t-SNE')
        plt.ylabel('Second t-SNE')
        plt.show()
        break

def graph_3dspace(net, test_dataloader, idx=0, seed=42):
    device = 'cuda:0'
    np.random.seed(seed)
    for center, context_negatives, masks, labels, feature in test_dataloader:
        center, context_negatives, masks, labels, feature = center.to(device), context_negatives.to(device), masks.to(device), labels.to(device), feature.to(device)
        embed_v = net[0](center, feature).cpu().detach().numpy()
        embed_u = net[1](context_negatives, feature).cpu().detach().numpy()

        # Stack the embeddings to run t-SNE on all of them together
        all_embeddings = np.vstack([embed_v[idx], embed_u[idx]])

        # Run t-SNE with n_components=3 for 3D visualization
        tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=seed)
        reduced_embeddings = tsne.fit_transform(all_embeddings)

        # Separate the reduced embeddings
        reduced_v = reduced_embeddings[0]
        reduced_u = reduced_embeddings[1:]

        # Plot in 3D
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
        ax.scatter(reduced_u[:, 0], reduced_u[:, 1], reduced_u[:, 2], c='blue', label='Context Vectors u')
        ax.scatter(reduced_v[0], reduced_v[1], reduced_v[2], c='red', label='Center Vector v')

        # Draw lines from v to the closest vectors in u
        distances = np.linalg.norm(reduced_u - reduced_v, axis=1)
        closest_indices = np.argsort(distances)[:5]  # Get the indices of the 5 closest vectors
        for index in closest_indices:
            ax.plot([reduced_v[0], reduced_u[index, 0]],
                    [reduced_v[1], reduced_u[index, 1]],
                    [reduced_v[2], reduced_u[index, 2]], 'grey', linestyle='--', alpha=0.5)

        ax.set_xlabel('First t-SNE')
        ax.set_ylabel('Second t-SNE')
        ax.set_zlabel('Third t-SNE')
        plt.legend()
        plt.show()
        break

if __name__ == "__main__":
    input_size = 4
    batch_size = 32
    max_length = 60
    multiplier_factor = 1
    embedding_size = 27

    # Load data
    dataset_path = get_config_path('data/CoopGrasping-v1_dataset.pkl')
    _, _, test_dataset, feature_structure = load_data_grasp(batch_size, multiplier_factor, dataset_path, max_length)

    # Load model
    net = nn.Sequential(
        CenterEmbeddingResNetv2(input_size, embedding_size, feature_structure), # v net --> centers
        ContextEmbeddingResNetv2(input_size, embedding_size, feature_structure)  # u net --> contexts
    )
    # net.load_state_dict(torch.load(get_config_path('weights/model_res_v2.pth'))) # model.pth
    net.load_state_dict(torch.load(get_config_path('../train/model_res_v2.pth'))) # model.pth
    net = net.to('cuda:0').eval()

    test_dataloader = DataLoader(test_dataset, batch_size=12, shuffle=True, collate_fn=batchify_wrapper(max_length))

    # Evaluate model
    evaluate_model(net, test_dataloader)
    evaluate_model_topk(net, test_dataloader)
