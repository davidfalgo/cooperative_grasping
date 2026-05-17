# Reorganized from train/TrainNS_train.py
# Original author:
# Description: Training loop, loss functions, validation routines, sweep configuration, and model factory

import os
import time
import yaml
import math
import torch
import wandb
import random
import pickle
import itertools
import numpy as np
from torch import nn
from tqdm import tqdm
from types import SimpleNamespace
from simulation.utils import *
from TrainNS_models import *
from torchinfo import summary
import torch.nn.functional as F
from TrainNS_dataloader import *
from simulation.utils import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
# np.random.seed(123)
# random.seed(123)
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, restore_best_weights=True, verbose=True):
        """
        Early stopping to terminate training when validation loss stops improving.

        Parameters:
            patience (int): Number of epochs with no improvement after which training stops.
            min_delta (float): Minimum decrease in the monitored metric to qualify as an improvement.
            restore_best_weights (bool): If True, restores model weights from the epoch with best performance.
            verbose (bool): Prints messages when improvement occurs.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_loss = float('inf')
        self.best_state_dict = None
        self.wait = 0
        self.stopped_epoch = 0

    def step(self, current_loss, model):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_state_dict = model.state_dict()
            self.wait = 0
            if self.verbose:
                print(f"[EarlyStopping] Improved validation loss: {current_loss:.5f}")
        else:
            self.wait += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement in validation loss for {self.wait} epochs.")
            if self.wait >= self.patience:
                self.stopped_epoch = True
                if self.verbose:
                    print("[EarlyStopping] Patience exceeded. Stopping training.")
                if self.restore_best_weights and self.best_state_dict:
                    model.load_state_dict(self.best_state_dict)
                return True
        return False


@torch.no_grad()
def evaluate_model(net, test_dataloader, device='cuda:0', threshold=None, verbose=True, cosine=True, method='weighted'):
    net.eval()
    all_preds, all_labels, all_masks = [], [], []

    for center, context_negatives, masks, labels, feature in test_dataloader:
        center, context_negatives, masks, labels, feature = [x.to(device) for x in [center, context_negatives, masks, labels, feature]]

        logits = prediction_temperature(center, context_negatives, net[0], net[1], feature, temperature=0.07)
        logits = logits.view_as(labels)

        # Apply masks
        valid_logits = logits[masks == 1]
        valid_labels = labels[masks == 1]

        all_preds.append(valid_logits.cpu())
        all_labels.append(valid_labels.cpu())

    all_preds = torch.cat(all_preds).sigmoid().numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Optimize threshold if not provided
    if threshold is None:
        thresholds = np.linspace(0, 1, 101)
        f1_scores = [f1_score(all_labels, all_preds >= t, average=method) for t in thresholds]
        best_idx = np.argmax(f1_scores)
        threshold = thresholds[best_idx]

    # Evaluate
    metrics = evaluate_metrics(all_labels, all_preds, threshold, verbose, method)

    precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
    pr_auc = auc(recall, precision)

    # if verbose:
    #     plot_roc_curve(all_labels, all_preds)
    #     plot_precision_recall_curve(all_labels, all_preds)
    #     plot_densities(all_labels, all_preds)

    return metrics, pr_auc, threshold

def evaluate_metrics(labels, preds, threshold=0.5, verbose=True, method='weighted'):
    binary_preds = (preds >= threshold).astype(int)

    accuracy  = accuracy_score(labels, binary_preds)
    precision = precision_score(labels, binary_preds, average=method, zero_division=0)
    recall    = recall_score(labels, binary_preds, average=method, zero_division=0)
    f1        = f1_score(labels, binary_preds, average=method, zero_division=0)
    auc_roc   = roc_auc_score(labels, preds)

    if verbose:
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
              f'Recall: {recall:.4f}, F1: {f1:.4f}, AUC-ROC: {auc_roc:.4f}, Threshold: {threshold:.2f}')

    return accuracy, precision, recall, f1, auc_roc

@torch.no_grad()
def evaluate_model_topk(net, test_dataloader, device='cuda:0', verbose=True, k_list=[1, 3, 5, 7], cosine=True):
    net.eval()
    top_k_correct = {k: 0 for k in k_list}
    total_samples = 0

    for center, context_negatives, masks, labels, feature in test_dataloader:
        center, context_negatives, masks, labels, feature = [x.to(device) for x in [center, context_negatives, masks, labels, feature]]

        logits = prediction_temperature(center, context_negatives, net[0], net[1], feature, temperature=0.07)
        logits[masks == 0] = -float('inf')

        for k in k_list:
            top_k_indices = logits.topk(k, dim=1).indices
            batch_hits = labels.gather(1, top_k_indices).any(dim=1).sum().item()
            top_k_correct[k] += batch_hits

        total_samples += (labels.sum(dim=1) > 0).sum().item()

    top_k_accuracy = {k: top_k_correct[k] / max(total_samples, 1) for k in k_list}

    if verbose:
        for k in k_list:
            print(f'Top-{k} Accuracy: {top_k_accuracy[k]:.4f}')

    return top_k_accuracy




def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

################################################################################### Fordward propagation and loss
def prediction(center, contexts_and_negatives, net_v, net_u, feature):
    v = net_v(center, feature)
    u = net_u(contexts_and_negatives, feature)
    # v_expanded = v.unsqueeze(1).expand(-1, 60, -1)
    # pred = F.cosine_similarity(v_expanded, u, dim=2)
    pred = torch.bmm(v.unsqueeze(1), u.permute(0, 2, 1))
    return pred.squeeze(1)

def prediction_temperature(center, contexts_and_negatives, net_v, net_u, feature, temperature=0.07):
    v = F.normalize(net_v(center, feature), dim=-1)
    u = F.normalize(net_u(contexts_and_negatives, feature), dim=-1)          # [B,K,E]
    pred = torch.bmm(v.unsqueeze(1), u.permute(0,2,1)) / temperature
    # pred = torch.sigmoid(logits)
    return pred.squeeze(1)

# Based on binary cross entropy loss
# class SigmoidBCELoss(torch.nn.Module):
#     def __init__(self, pos_weight=1.0, neg_weight=1.0):
#         super(SigmoidBCELoss, self).__init__()
#         self.pos_weight = pos_weight
#         self.neg_weight = neg_weight

#     def forward(self, pred, label, mask=None):
#         label   = label.type_as(pred)
#         weights = label * self.pos_weight + (1 - label) * self.neg_weight
#         if mask is not None:
#             weights *= mask
#         loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label, weight=weights, reduction='none')
#         loss = (loss * weights.ne(0)).sum(-1) / (weights.ne(0).sum(-1) + 1e-9)
#         return loss #loss.mean(dim=1)

#################################################################################### Training loop
def validate_original(net, validation_dataloader, loss, device='cuda:0', verbose=False, k=2):
    loop = tqdm(enumerate(validation_dataloader), total=len(validation_dataloader), leave=True)
    net.train(False)
    total_loss = 0
    all_preds = []
    all_labels = []
    top_k_correct = 0
    total_samples = 0
    for i, batch in loop:
        center, context_negative, mask, label, feature = batch
        center, context_negative, mask, label, feature = center.to(device), context_negative.to(device), mask.to(device), label.to(device), feature.to(device)
        pred = prediction_temperature(center, context_negative, net[0], net[1], feature, 0.07)
        # pred = prediction(center, context_negative, net[0], net[1], feature)
        l = (loss(pred.reshape(label.shape).float(), label.float(), mask) / mask.sum(axis=1)*mask.shape[1])
        total_loss += l.sum().item()
        if verbose:
            loop.set_description(f'Validation, Loss: {l.sum().item():.4f}')
            loop.set_postfix(loss=l.sum().item())
        all_preds.append(pred)
        all_labels.append(label)

        # Evaluate top-k metrics
        # _, top_k_indices = (pred*mask).topk(k, dim=1) # ORIGINAL
        masked_pred = pred.clone()
        masked_pred[mask == 0] = -float('inf')
        _, top_k_indices = masked_pred.topk(k, dim=1)
        for batch_item in range(label.size(0)):  # iterate over the batch
            true_label_indices = label[batch_item].nonzero(as_tuple=True)[0].tolist()
            top_k_indices_batch = top_k_indices[batch_item].tolist()
            if any(idx in top_k_indices_batch for idx in true_label_indices):
                top_k_correct += 1
        # total_samples += label.size(0) # ORIGINAL
        total_samples += (label.sum(dim=1) > 0).sum().item()

    # Calculate top-k accuracy
    top_k_accuracy = top_k_correct / total_samples

    all_preds = torch.cat(all_preds, dim=0).cpu().detach().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().detach().numpy()
    acc, f1 = test_metrics(all_preds, all_labels)

    return total_loss / len(validation_dataloader), acc, f1, top_k_accuracy

# --------------------------------------------------------------------- 4.5
# Robust binary cross entropy loss with masking
class SigmoidBCELoss(torch.nn.Module):
    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        super(SigmoidBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, pred, label, mask=None):
        label = label.float()
        weights = label * self.pos_weight + (1 - label) * self.neg_weight
        if mask is not None:
            weights *= mask.float()
        loss = F.binary_cross_entropy_with_logits(pred, label, weight=weights, reduction='none')
        return loss.sum() / weights.sum()

# Robust binary cross entropy loss with masking
class DynamicSigmoidBCELoss(torch.nn.Module):
    def __init__(self):
        super(DynamicSigmoidBCELoss, self).__init__()

    def forward(self, pred, label, mask=None):
            label = label.float()
            # Dynamically compute weights
            pos_counts = (label * mask.float()).sum(dim=1, keepdim=True)  # [batch, 1]
            neg_counts = ((1 - label) * mask.float()).sum(dim=1, keepdim=True)  # Count negatives per center
            # neg_counts = torch.clamp(neg_counts, min=1e-8)  # Avoid division by zero
            neg_weights = pos_counts / neg_counts  # Weight = |DC|
            neg_weights = neg_weights.repeat(1, label.size(1))  # Replicate for each sample
            weights = label + (1 - label) * neg_weights  # Positive=1, Negative=|DC|/|N|

            if mask is not None:
                weights *= mask.float()

            loss = F.binary_cross_entropy_with_logits(pred, label, weight=weights, reduction='none')
            return loss.sum()/ label.shape[0] #/ weights.sum()

class DynamicWeightSigmoidBCELoss(torch.nn.Module):
    def __init__(self, beta=0.5):
        super(DynamicWeightSigmoidBCELoss, self).__init__()
        self.beta = beta  # Weighting factor for positive/negative samples

    def forward(self, pred, label, mask=None):
            label = label.float()
            # Dynamically compute weights
            pos_counts = (label * mask.float()).sum(dim=1, keepdim=True)  # [batch, 1]
            neg_counts = ((1 - label) * mask.float()).sum(dim=1, keepdim=True)  # Count negatives per center
            neg_counts = torch.clamp(neg_counts, min=1e-8)  # Avoid division by zero
            neg_weights = pos_counts
            # neg_weights = (pos_counts / neg_counts)#*self.beta  # Weight = |DC|
            neg_weights = neg_weights.repeat(1, label.size(1))  # Replicate for each sample
            weights = label + (1 - label) * neg_weights  # Positive=1, Negative=|DC|/|N|

            if mask is not None:
                weights *= mask.float()

            loss = F.binary_cross_entropy_with_logits(pred, label, weight=weights, reduction='none')
            return loss.sum()/ label.shape[0] #/ weights.sum()


# Improved validation function
@torch.no_grad()
def validate(net, validation_dataloader, loss_fn, device='cuda:0', verbose=False, k=2):
    net.eval()
    total_loss, total_pairs = 0.0, 0
    all_preds, all_labels = [], []
    top_k_correct, total_samples = 0, 0

    loop = tqdm(validation_dataloader, desc="Validation", leave=verbose)
    for batch in loop:
        center, context_negative, mask, label, feature = [x.to(device) for x in batch]

        logits = prediction_temperature(center, context_negative, net[0], net[1], feature, temperature=0.07)
        logits = logits.view_as(label)

        # Proper masked loss calculation
        batch_loss = loss_fn(logits, label, mask)
        valid_pairs = mask.sum().item()
        total_loss += batch_loss.item() * valid_pairs
        total_pairs += valid_pairs

        # Store for metrics calculation
        probs = logits.sigmoid().detach().cpu()
        all_preds.append(probs)
        all_labels.append(label.cpu())

        # Efficient Top-K recommendation calculation
        logits[mask == 0] = -float('inf')
        top_k_indices = logits.topk(k, dim=1).indices
        positives = label.bool()
        hits = positives.gather(1, top_k_indices).any(dim=1).sum().item()

        top_k_correct += hits
        total_samples += (label.sum(dim=1) > 0).sum().item()

        if verbose:
            loop.set_postfix(loss=batch_loss.item())

    avg_loss = total_loss / total_pairs
    hit_at_k = top_k_correct / max(total_samples, 1)

    # Threshold-tuned F1 and accuracy
    all_preds = torch.cat(all_preds).numpy().ravel()
    all_labels = torch.cat(all_labels).numpy().ravel()

    # thresholds = np.linspace(0, 1, 50)
    # f1_scores = [f1_score(all_labels, all_preds >= t, average='weighted') for t in thresholds]
    # best_idx = np.argmax(f1_scores)
    # best_thr = thresholds[best_idx]
    best_thr = 0.5  # Default threshold if dynamic search is not used
    # best_f1 = f1_scores[best_idx]
    best_f1 = f1_score(all_labels, all_preds >= best_thr, average='weighted')
    best_acc = accuracy_score(all_labels, all_preds >= best_thr)
    precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
    pr_auc = auc(recall, precision)
    # print(f'PR AUC: {pr_auc:.4f}')

    if verbose:
        print(f"\n[VAL] Loss: {avg_loss:.4f}, Acc: {best_acc:.4f}, F1: {best_f1:.4f}, "
              f"Hit@{k}: {hit_at_k:.4f}, Optimal Threshold: {best_thr:.2f}, PR AUC: {pr_auc:.4f}")

    return avg_loss, best_acc, best_f1, hit_at_k, best_thr, pr_auc

# --------------------------------------------------------------------- o3
@torch.no_grad()
def validate_o3(net,
             dataloader,
             criterion,
             device="cuda:0",
             k=2,
             verbose=False):
    """
    Validation loop with:
      • model.eval()  +  no_grad()
      • per-pair masked BCE loss, averaged over *valid* pairs
      • dynamic threshold search for best F1 / Accuracy
      • vectorised Hit@k (a.k.a. Recall@k)
    Returns
      avg_loss, accuracy, f1, hit@k, best_threshold
    """
    net.eval()

    running_loss, valid_pairs = 0.0, 0
    all_logits, all_labels    = [], []
    hitk_num, hitk_den        = 0, 0

    loop = tqdm(dataloader, total=len(dataloader), leave=verbose)
    for center, ctx_neg, mask, label, feat in loop:
        center   = center.to(device)
        ctx_neg  = ctx_neg.to(device)
        mask     = mask.to(device)
        label    = label.to(device)
        feat     = feat.to(device)

        # -------- forward pass (logits) -------------------------------
        logits = prediction_temperature(center, ctx_neg,
                                        net[0], net[1], feat, temperature=0.07)
        logits = logits.view_as(label)                     # (B, K)

        # -------- loss over valid pairs -------------------------------
        loss_vec   = criterion(logits, label, mask)        # (B,)
        running_loss += loss_vec.sum().item()
        valid_pairs  += mask.sum().item()

        # -------- accumulate for metrics ------------------------------
        all_logits.append(logits.cpu())
        all_labels.append(label.cpu())

        # -------- Hit@k / top-k ---------------------------------------
        logits_masked = logits.clone()
        logits_masked[mask == 0] = -float("inf")
        topk_idx = logits_masked.topk(k, dim=1).indices    # (B, k)

        hit = label.bool().gather(1, topk_idx).any(dim=1)  # (B,)
        hitk_num += hit.sum().item()
        hitk_den += (label.sum(dim=1) > 0).sum().item()    # anchors w/ ≥1 positive

    # -----------------------------------------------------------------
    avg_loss = running_loss / max(valid_pairs, 1)

    logits = torch.cat(all_logits).sigmoid().numpy().ravel()
    labels = torch.cat(all_labels).numpy().ravel()

    # dynamic threshold search (0–1 in steps of 0.01)
    thresholds = np.linspace(0, 1, 101)
    f1_scores  = [f1_score(labels, logits >= t) for t in thresholds]
    best_idx   = int(np.argmax(f1_scores))
    best_thr   = thresholds[best_idx]

    preds_bin  = logits >= best_thr
    f1         = f1_scores[best_idx]
    acc        = accuracy_score(labels, preds_bin)

    hit_k = hitk_num / max(hitk_den, 1)

    if verbose:
        print(f"[VAL] loss={avg_loss:.4f}  acc={acc:.3f}  f1={f1:.3f}  "
              f"hit@{k}={hit_k:.3f}  thr={best_thr:.2f}")

    return avg_loss, acc, f1, hit_k, best_thr

# # Based on binary cross entropy loss
# class SigmoidBCELoss(torch.nn.Module):
#     def __init__(self, pos_weight=1.0, neg_weight=1.0):
#         super(SigmoidBCELoss, self).__init__()
#         self.pos_weight = pos_weight
#         self.neg_weight = neg_weight

#     def forward(self, pred, label, mask=None):
#         label   = label.type_as(pred)
#         weights = label * self.pos_weight + (1 - label) * self.neg_weight
#         if mask is not None:
#             weights *= mask
#         loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label, weight=weights, reduction='none')
#         loss = (loss * weights.ne(0)).sum(-1) / (weights.ne(0).sum(-1) + 1e-9)
#         return loss #loss.mean(dim=1)
# ---------------------------------------------------------------------

@torch.enable_grad()
def train_from_sweep(net, train_dataloader, validation_dataloader, config, device='cuda:0', model_path='model.pth'):
    """
    Improved training loop consistent with updated validation logic, dynamic threshold tuning, and stable loss calculations.
    """

    # Extract hyperparameters
    lr = config.learning_rate
    num_epochs = config.epochs
    positive_weight = config.positive_weight
    negative_weight = config.negative_weight
    lr_scheduler_patience = config.lr_scheduler_patience
    lr_scheduler_step = config.lr_scheduler_step

    # Move network to device
    net = net.to(device)

    # Setup optimizer and LR scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=lr_scheduler_patience,
        factor=lr_scheduler_step, verbose=True
    )

    # Initialize loss function with proper masking
    # loss_fn = SigmoidBCELoss(positive_weight, negative_weight)
    loss_fn = DynamicSigmoidBCELoss()

    # Initialize W&B
    wandb.init(project='Grasp-Negative-Sampling', entity='david-alvear', config=config)
    wandb.watch(net, loss_fn, log='all')

    best_val_loss = float('inf')
    train_loss_hist, val_loss_hist = [], []

    # Training loop
    for epoch in range(num_epochs):
        net.train()
        loop = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

        epoch_loss, total_pairs, start_time = 0.0, 0, time.time()
        all_preds, all_labels = [], []

        for batch_idx, batch in enumerate(loop):
            center, context_negative, mask, label, feature = [x.to(device) for x in batch]

            # Forward pass
            pred = prediction_temperature(center, context_negative, net[0], net[1], feature, temperature=0.07)
            pred = pred.view_as(label)

            # Compute masked loss
            loss_batch = loss_fn(pred, label, mask)

            # Backpropagation
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            valid_pairs = mask.sum().item()
            epoch_loss += loss_batch.item() * valid_pairs
            total_pairs += valid_pairs

            # Store for metrics
            all_preds.append(pred.detach().cpu())
            all_labels.append(label.detach().cpu())

            if batch_idx % 100 == 0:
                loop.set_postfix(loss=loss_batch.item())
                wandb.log({'train/batch_loss': loss_batch.item()})

        # Epoch metrics
        epoch_loss /= total_pairs
        total_time = time.time() - start_time

        all_preds = torch.cat(all_preds).sigmoid().numpy().ravel()
        all_labels = torch.cat(all_labels).numpy().ravel()

        thresholds = np.linspace(0, 1, 50)
        f1_scores = [f1_score(all_labels, all_preds >= t, average='weighted') for t in thresholds]
        best_idx = np.argmax(f1_scores)
        best_thr_train = thresholds[best_idx]

        acc_train = accuracy_score(all_labels, all_preds >= best_thr_train)
        f1_train = f1_scores[best_idx]

        # Validation step
        val_loss, val_acc, val_f1, val_hit_k, best_thr_val, pr_auc = validate(
            net, validation_dataloader, loss_fn, device, k=1, verbose=True
        )

        scheduler.step(val_loss)

        # Log metrics
        wandb.log({
            'train/epoch_loss': epoch_loss,
            'train/acc': acc_train,
            'train/f1': f1_train,
            'train/best_threshold': best_thr_train,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'val/f1': val_f1,
            'val/topk': val_hit_k,
            'val/best_threshold': best_thr_val,
            'epoch': epoch,
            'pairs/s': total_pairs / total_time,
            'val/pr_auc': pr_auc,
        })

        train_loss_hist.append(epoch_loss)
        val_loss_hist.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), model_path)
            wandb.log({"best_val_loss": best_val_loss})

        if val_hit_k > 0.837:
            print(f"Hit@{1} threshold reached: {val_hit_k:.4f}, saving model.")
            torch.save(net.state_dict(), model_path)
            wandb.log({"hit_threshold_reached": val_hit_k})
            break

        print(f"Epoch [{epoch+1}/{num_epochs}] finished. Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, F1-val: {val_f1:.4f}")

    wandb.finish()

    return net, {'train_loss': train_loss_hist, 'val_loss': val_loss_hist}

def train(net, train_dataloader:DataLoader, validation_dataloader, lr, num_epochs, l1_lambda=0.001, device='cuda:0', model_path='model.pth'):
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)
    loss = SigmoidBCELoss(10.0, 1.0)

    # Initialize Weights & Biases
    wandb.init(project='Grasp-Negative-Sampling', entity='david-alvear')
    wandb.config.update({
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": train_dataloader.batch_size,
        "optimizer": "Adam",
        "loss": "BCELoss",
        "Net": "GeometricEmbeddingResNet",
        "loss_weight_positive": 10.0,
        "loss_weight_negative": 1.0,
        "scheduler": "ReduceLROnPlateau",
        "grasping_function": "mean",
        "map": True,
        # Add other hyperparameters or settings here
    })
    wandb.watch(net, loss, log='all')
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True)
        total_loss = 0
        tokens = 0
        t0 = time.time()
        net.train(True)
        for i, batch in loop:
            optimizer.zero_grad()
            center, context_negative, mask, label, feature = batch # TODO: features not torch tensor
            center, context_negative, mask, label, feature = center.to(device), context_negative.to(device), mask.to(device), label.to(device), feature.to(device)

            pred = prediction_temperature(center, context_negative, net[0], net[1], feature, 0.07)
            # pred = prediction(center, context_negative, net[0], net[1], feature)
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask) / mask.sum(axis=1)*mask.shape[1])
            # l1_loss = sum(param.abs().sum() for param in net.parameters())
            loss_total = l.sum() #+ l1_lambda * l1_loss
            loss_total.backward()
            optimizer.step()

            total_loss += l.sum().item()
            tokens += l.numel()

            if i % 20 == 0:
                loss_step = l.sum().item()
                wandb.log({'train/loss': loss_step})
                loop.set_description(f'Epoch [{epoch}/{num_epochs}], Loss: {loss_step:.4f}')
                loop.set_postfix(loss=loss_step)

        total_time = time.time() - t0
        epoch_loss = total_loss / len(train_dataloader)
        val_loss, acc, f1, top_k_accuracy = validate(net, validation_dataloader, loss, device)
        scheduler.step(val_loss)
        wandb.log({'train/epoch_loss': epoch_loss, 'epoch': epoch, 'tokens/s': tokens / total_time, 'val/topk': top_k_accuracy,
            'val/loss': val_loss, 'val/acc': acc, 'val/f1': f1})

        # wandb.log({'train/epoch_loss': epoch_loss, 'epoch': epoch, 'tokens/s': tokens / total_time, 'val/loss': val_loss})
        # print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}, Tokens/s: {tokens /total_time:.4f}')

        # Checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), model_path)
            wandb.log({"best_val_loss": best_val_loss})

    # torch.save(net.state_dict(), model_path)
    wandb.save(model_path)
    wandb.finish()

def get_similar_vectors(query_vector, contexts_vectors, net_v, net_u, feature, cosine=False):
    # Generate embedding for the query vector using net_v
    v_embedding = net_v(query_vector, feature)  # Add batch dimension to query_vector if needed

    # Generate embeddings for context vectors using net_u
    u_embeddings = net_u(contexts_vectors, feature)  # Assume contexts_vectors is batched if necessary

    # Calculate cosine similarity between v_embedding and each of u_embeddings
    if cosine:
        cos_similarities = F.cosine_similarity(v_embedding.unsqueeze(dim=1), u_embeddings, dim=2)
    else:
        cos_similarities = torch.bmm(v_embedding.unsqueeze(1), u_embeddings.permute(0, 2, 1)).squeeze(1)

    return cos_similarities

def test_metrics(pred, labels, threshold_pos=0.55):
    # labels = all_labels.cpu().detach().numpy()
    # pred = all_preds.cpu().detach().numpy()
    # normalized_pred = (pred + 1) / 2      # ORIGINAL
    pred_clipped = np.clip(pred, -60.0, 60.0)
    normalized_pred = 1 / (1 + np.exp(-pred_clipped))
    binarized_pred = (normalized_pred > threshold_pos).astype(int)

    # Calculate metrics
    accuracy = (labels == binarized_pred).mean()
    f1 = f1_score(labels.flatten(), binarized_pred.flatten(), average='weighted', zero_division=0)
    return accuracy, f1

#################################################################################### Sweep

def sweep_weights_bias(config=None):
    with wandb.init(project='Grasp-Negative-Sampling', entity='david-alvear', config=config):
        config = wandb.config

        multiplier_factor = config.multiplier_factor
        device = config.device
        lr = config.learning_rate
        num_epochs = config.epochs
        input_size = config.input_size
        max_length = config.max_length
        batch_size = config.batch_size
        embedding_size = config.embedding_size
        negative_weight = config.negative_weight
        positive_weight = config.positive_weight

        #new items
        lr_scheduler_patience = config.lr_scheduler_patience
        lr_scheduler_step = config.lr_scheduler_step
        dropout = config.dropout
        cograsp_dataset = config.dataset
        model_path = config.model_path
        temperature = config.temperature
        model_v = config.model_version
        weight_decay = config.weight_decay

        set_seed(config.seed)

        dataset_path = get_config_path(f'data/{cograsp_dataset}') # CoopGrasping-v5_dataset.pkl
        train_dataloader, validation_dataloader, test_dataset, feature_structure = load_data_grasp(batch_size, multiplier_factor, dataset_path, max_length)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=batchify_wrapper(config.max_length))

        net = define_model(input_size, embedding_size, feature_structure, dropout, config.map, device, model_version=model_v)

        # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)  # AdamW for better generalization
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=lr_scheduler_patience, factor=lr_scheduler_step, verbose=True)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)
        # loss_fn = SigmoidBCELoss(positive_weight, negative_weight)
        loss_fn = DynamicSigmoidBCELoss()  # Use the dynamic loss function
        # loss_fn = DynamicWeightSigmoidBCELoss(beta=positive_weight)  # Use the dynamic loss function with beta
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        max_val_topk = 0.0

        best_val_loss = float('inf')
        wandb.watch(net, loss_fn, log='all')
        for epoch in range(num_epochs):
            net.train()

            loop = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
            epoch_loss, total_pairs, start_time = 0.0, 0, time.time()
            all_preds, all_labels = [], []

            for batch_idx, batch in enumerate(loop):
                optimizer.zero_grad()
                center, context_negative, mask, label, feature = [x.to(device) for x in batch]

                # Forward pass
                pred = prediction_temperature(center, context_negative, net[0], net[1], feature, temperature)
                pred = pred.view_as(label)
                # pred = prediction(center, context_negative, net[0], net[1], feature)

                # compute masked loss
                loss_batch = loss_fn(pred, label, mask)

                # backpropagation
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()

                valid_pairs = mask.sum().item()
                epoch_loss += loss_batch.item() * valid_pairs
                total_pairs += valid_pairs

                # Store for metrics
                all_preds.append(pred.detach().cpu())
                all_labels.append(label.detach().cpu())

                if batch_idx % 100 == 0:
                    loop.set_postfix(loss=loss_batch.item())
                    wandb.log({'train/batch_loss': loss_batch.item()})


            # Epoch metrics
            epoch_loss /= total_pairs
            total_time = time.time() - start_time

            all_preds = torch.cat(all_preds).sigmoid().numpy().ravel()
            all_labels = torch.cat(all_labels).numpy().ravel()

            thresholds = np.linspace(0, 1, 50)
            f1_scores = [f1_score(all_labels, all_preds >= t, average='weighted') for t in thresholds]
            best_idx = np.argmax(f1_scores)
            best_thr_train = thresholds[best_idx]

            acc_train = accuracy_score(all_labels, all_preds >= best_thr_train)
            f1_train = f1_scores[best_idx]

            # Validation
            val_loss, val_acc, val_f1, val_hit_k, best_thr_val, pr_auc = validate(
                net, validation_dataloader, loss_fn, device, k=1, verbose=True
            )

            scheduler.step(val_loss)

            if val_hit_k > max_val_topk:
                max_val_topk = val_hit_k

            # Log metrics
            wandb.log({
                'train/epoch_loss': epoch_loss,
                'train/acc': acc_train,
                'train/f1': f1_train,
                'train/best_threshold': best_thr_train,
                'val/loss': val_loss,
                'val/acc': val_acc,
                'val/f1': val_f1,
                'val/topk': val_hit_k,
                'val/best_threshold': best_thr_val,
                'epoch': epoch,
                'pairs/s': total_pairs / total_time,
                'val/pr_auc': pr_auc,
                'max_val_topk': max_val_topk
            })
            # Checkpoint saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # torch.save(net.state_dict(), model_path)
                wandb.log({"best_val_loss": best_val_loss})
            print(f"Epoch [{epoch+1}/{num_epochs}] finished. Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, F1-val: {val_f1:.4f}")

            if val_hit_k > 0.89:
                model_path_topk = model_path + f'{model_v}_E{embedding_size}_B{batch_size}_topk_{val_hit_k:.3f}.pth'
                torch.save(net.state_dict(), model_path_topk)

            if early_stopping.step(val_loss, net):
                print(f"Stopped early at epoch {epoch+1}")
                break

        # use the best model for testing
        # Evaluate with optimal threshold search and default (weighted) metrics
        net = net.to(device).eval()
        metrics, pr_auc, best_threshold = evaluate_model(
            net,
            test_dataloader,
            device=device,      # or your device
            threshold=None,       # will search for the F1-optimal threshold
            verbose=True,
            method='weighted'
        )
        topk_accuracies = evaluate_model_topk(
            net,
            test_dataloader,
            device=device,
            verbose=True,   # print the results
            k_list=[1, 3, 5]
        )
        # The result
        accuracy, precision, recall, f1, auc_roc = metrics
        wandb.log({
            'test/accuracy': accuracy,
            'test/precision': precision,
            'test/recall': recall,
            'test/f1': f1,
            'test/auc_roc': auc_roc,
            'test/best_threshold': best_threshold,
            'test/topk_1': topk_accuracies[1],
            'test/pr_auc': pr_auc,
        })

def define_model(input_size, embedding_size, feature_structure, dropout, map_flag=True, device='cuda:0', model_version='v3'):
    if map_flag:
        if model_version == 'v2':
            net = nn.Sequential(
                CenterEmbeddingResNetv2(input_size, embedding_size, feature_structure, dropout), # v net --> centers
                ContextEmbeddingResNetv2(input_size, embedding_size, feature_structure, dropout)  # u net --> contexts
            )
        elif model_version == 'v3':
            net = nn.Sequential(
                CenterEmbeddingResNetv3(input_size, embedding_size, feature_structure, dropout), # v net --> centers
                ContextEmbeddingResNetv3(input_size, embedding_size, feature_structure, dropout)  # u net --> contexts
            )
        elif model_version == 'v4':
            net = nn.Sequential(
                CenterEmbeddingResNetv4(input_size, embedding_size, feature_structure, dropout), # v net --> centers
                ContextEmbeddingResNetv4(input_size, embedding_size, feature_structure, dropout)  # u net --> contexts
            )
    else:
        net = nn.Sequential(
            CenterEmbeddingResidual(input_size, embedding_size, feature_structure), # v net --> centers
            ContextEmbeddingResidual(input_size, embedding_size, feature_structure)  # u net --> contexts
        )
    return net.to(device)

#################################################################################### Main
def sweep():
    sweep_config = {
        'method': 'bayes',  # or 'bayes', 'grid', 'random' etc.
        'metric': {'goal': 'maximize', 'name': 'test/pr_auc'},# val/topk, val/loss, pr_auc
        # FOR SWEEPING ++++++++++++++++++++++++++
        'parameters': {
            'batch_size': {'distribution': 'int_uniform', 'max': 80, 'min': 16},
            'epochs': {'value': 80},
            'embedding_size': {'distribution': 'int_uniform', 'min': 20, 'max': 100},
            'learning_rate': {'distribution': 'uniform', 'min': 0.0001, 'max': 0.0015},# 0.009
            # 'learning_rate': {'value': 0.00040249488812477743},# 0.009
            'lr_scheduler_patience': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
            'lr_scheduler_step': {'distribution': 'uniform', 'min': 0.01, 'max': 0.4},
            'l1_lambda': {'value': 0.001},
            'input_size': {'value': 4},
            'dropout': {'value': 0.02},
            # 'dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.5},
            'dataset': {'value': 'CoopGrasping-v6_dataset.pkl'}, # CoopGrasping-v2_dataset.pkl
            # 'multiplier_factor': {'distribution': 'uniform', 'min': 1.0, 'max': 3.3},  # negative sampling factor 1.0 = total_lenght = context
            'multiplier_factor': {'value': 1.1},  # negative sampling factor 1.0 = total_lenght = context
            'max_length': {'value': 100},
            'device': {'value': 'cuda:0'},
            'random': {'value': True},
            'negative_weight': {'value': 1.0},
            'positive_weight': {'value': 1.0},
            # 'positive_weight': {'distribution': 'int_uniform', 'min': 1, 'max': 12},
            'map': {'value': True},
            'seed': {'value': 1234},
            'model_path': {'value': '/ibex/scratch/alveardf/slurm_outputs/weights/swmodel_'},  # Path to save the model
            'temperature': {'distribution': 'uniform', 'min': 0.02, 'max': 0.08},  # Temperature for softmax
            # 'temperature': {'value': 0.07},  # Temperature for softmax
            'model_version': {'value': 'v4'},  # Model version for naming.  'v2', 'v3', 'v4' etc.
            'loss': {'value': 'DynamicSigmoidBCELoss'},  # Loss function to use
            'weight_decay': {'value': 1e-4},  # Weight decay for AdamW optimizer
        }
        # FOR TRAINING ++++++++++++++++++++++++++
        # 'parameters': {
        #     'batch_size': {'value': 19},
        #     'epochs': {'value': 100},
        #     'embedding_size': {'value': 91},
        #     'learning_rate': {'distribution': 'uniform', 'min': 0.00001, 'max': 0.005},# 0.009
        #     # 'learning_rate': {'value': 0.00040249488812477743},# 0.009
        #     'lr_scheduler_patience': {'value':  5},
        #     'lr_scheduler_step': {'value': 0.4619360566709294},
        #     'l1_lambda': {'value': 0.001},
        #     'input_size': {'value': 4},
        #     'dropout': {'value': 0.067},
        #     'dataset': {'value': 'CoopGrasping-v6_dataset.pkl'}, # CoopGrasping-v2_dataset.pkl
        #     'multiplier_factor': {'value': 1.0},  # negative sampling factor 1.0 = total_lenght = context
        #     # 'multiplier_factor': {'value': 2.1},  # negative sampling factor 1.0 = total_lenght = context
        #     'max_length': {'value': 100},
        #     'device': {'value': 'cuda:0'},
        #     'random': {'value': True},
        #     'negative_weight': {'value': 1.0},
        #     # 'positive_weight': {'value': 1.0},
        #     'positive_weight': {'distribution': 'uniform', 'min': 0.5, 'max': 12},
        #     'map': {'value': True},
        #     'seed': {'value': 1234},
        #     'model_path': {'value': '/ibex/scratch/alveardf/slurm_outputs/weights/swmodel_'},  # Path to save the model
        #     # 'temperature': {'distribution': 'uniform', 'min': 0.01, 'max': 0.1},  # Temperature for softmax
        #     'temperature': {'value': 0.07},  # Temperature for softmax
        #     'model_version': {'value': 'v3'},  # Model version for naming.  'v2', 'v3', 'v4' etc.
        # }
    }

    # Initialize a new sweep
    sweep_id = wandb.sweep(sweep_config, project='Grasp-Negative-Sampling', entity='david-alvear')
    # Sweep the weights and bias
    wandb.agent(sweep_id, function=sweep_weights_bias, count=350) # count=1, 350

def train_main(input_size, batch_size, max_length, embedding_size, multiplier_factor, lr, num_epochs, dataset_path, model_path):
    train_dataloader, validation_dataloader, test_dataset, feature_structure = load_data_grasp(batch_size, multiplier_factor, dataset_path, max_length)

    # Resnet model
    net = nn.Sequential(
        CenterEmbeddingResNetv2(input_size, embedding_size, feature_structure), # v net --> centers
        ContextEmbeddingResNetv2(input_size, embedding_size, feature_structure)  # u net --> contexts
    )
    # Training
    train(net, train_dataloader, validation_dataloader, lr, num_epochs)
    # Evaluation


########################################## Plots
def plot_training_history(history, size=(8,6), title_size=18, font_size=14, tick_size=12, linewidth=3):
    """
    Plots the training and validation loss curves from a training history dictionary.

    Parameters:
    -----------
    history : dict
        A dictionary containing the epoch-wise training and validation losses.
    size : tuple
        The figure size in inches (width, height).
    title_size : int
        The font size for the plot title.
    font_size : int
        The font size for the x/y labels and legend.
    tick_size : int
        The font size for the x/y ticks.

    Usage:
    ------
    plot_training_history(history)
    """

    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])

    if not train_loss or not val_loss:
        print("Error: 'train/loss' or 'val/loss' is missing or empty in the history dictionary.")
        return

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=size)
    plt.plot(epochs, train_loss, label='Train Loss', linewidth=linewidth)
    plt.plot(epochs, val_loss, label='Validation Loss', linewidth=linewidth)

    # Axis labels
    plt.xlabel('Epoch', fontsize=font_size)
    plt.ylabel('Loss', fontsize=font_size)

    # Title
    plt.title('Training vs Validation Loss', fontsize=title_size)

    # Set tick font sizes
    # plt.xticks(np.linspace(1, len(train_loss), num=26, dtype=int), fontsize=tick_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    # Legend
    plt.legend(fontsize=font_size)

    plt.show()

if __name__ == "__main__":
    # lr, num_epochs = 0.001, 20
    # #################################################################################### Load dataloader and dataset
    # input_size = 4
    # batch_size = 64
    # max_length = 60
    # embedding_size = 100
    # multiplier_factor = 1
    # lr, num_epochs = 0.001, 20
    # dataset_path = get_config_path('data/CoopGrasping-v6_dataset.pkl')

    # config = SimpleNamespace(
    #     batch_size=19,
    #     embedding_size=83,  # 21
    #     learning_rate=0.00038444533213018547,
    #     lr_scheduler_patience=5,
    #     lr_scheduler_step=0.4619360566709294,
    #     positive_weight=1.0,
    #     negative_weight=1.0,
    #     epochs=180, # 91
    #     input_size=4,
    #     dropout=0.06757083710115819,
    #     max_length=100,
    #     multiplier_factor=1.0179277057095852,
    #     device='cuda:0',
    #     seed=1234,
    #     mixed_precision=False,  # Enable mixed precision training
    # )
    # project_root = os.path.dirname(os.path.abspath(__file__))
    # model_name = os.path.join(project_root, f'modelv4_e{config.epochs}|d{config.embedding_size}|lr{round(config.learning_rate,5)}|.pth')
    # dataset_path = os.path.join(project_root, '../config/data/CoopGrasping-v6_dataset.pkl')
    # train_dataloader, validation_dataloader, test_dataset, feature_structure = load_data_grasp(config.batch_size, config.multiplier_factor, dataset_path, config.max_length)

    # # model
    # net = nn.Sequential(
    #     CenterEmbeddingResNetv3(config.input_size, config.embedding_size, feature_structure, config.dropout), # v net --> centers
    #     ContextEmbeddingResNetv3(config.input_size, config.embedding_size, feature_structure, config.dropout)  # u net --> contexts
    # )

    # net, history = train_from_sweep(net, train_dataloader, validation_dataloader, config, device='cuda:0', model_path=model_name)
    # train_main(input_size, batch_size, max_length, embedding_size, multiplier_factor, lr, num_epochs, dataset_path, '../config/weights/model_v2.pth')

    # Uncomment to run the sweep
    sweep()
