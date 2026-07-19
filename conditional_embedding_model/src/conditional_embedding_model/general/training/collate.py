# Description: Collate functions producing the Trainer batch contract
# `((center_inputs, context_inputs), label, mask)`, plus an adapter for the frozen
# legacy `batchify_wrapper` 5-tuple output.

import random
import torch


def make_collate(max_length, index_key="index", feature_key="feature"):
    """Collate raw (center, context, negative, feature) dataset items into the
    Trainer batch contract. Index tensors are created as dtype=torch.long directly
    (the proper fix for the legacy `x.long()` FIXME)."""

    def collate(batch):
        centers, contexts_negatives, masks, labels, features = [], [], [], [], []
        for center, context, negative, feature in batch:
            combined = list(context) + list(negative)
            combined_labels = [1] * len(context) + [0] * len(negative)

            paired = list(zip(combined, combined_labels))
            random.shuffle(paired)
            shuffled_combined, shuffled_labels = zip(*paired) if paired else ((), ())

            cur_len = len(shuffled_combined)
            if cur_len > max_length:
                raise ValueError(f"context+negative length {cur_len} exceeds max_length {max_length}")

            centers.append([center])
            contexts_negatives.append(list(shuffled_combined) + [0] * (max_length - cur_len))
            masks.append([1] * cur_len + [0] * (max_length - cur_len))
            labels.append(list(shuffled_labels) + [0] * (max_length - cur_len))
            features.append(feature)

        center_inputs = {
            index_key: torch.tensor(centers, dtype=torch.long),
            feature_key: torch.tensor(features, dtype=torch.float),
        }
        context_inputs = {
            index_key: torch.tensor(contexts_negatives, dtype=torch.long),
            feature_key: torch.tensor(features, dtype=torch.float),
        }
        label = torch.tensor(labels, dtype=torch.float)
        mask = torch.tensor(masks, dtype=torch.float)
        return (center_inputs, context_inputs), label, mask

    return collate


class LegacyBatchAdapter:
    """Converts the frozen `batchify_wrapper` 5-tuple
    (center, contexts_negatives, mask, label, feature) into the Trainer batch
    contract, with long index tensors. Usable either as an iterable wrapper around
    an existing DataLoader, or as a collate post-processor called with the 5-tuple.
    """

    def __init__(self, loader=None):
        self.loader = loader

    @staticmethod
    def _convert(center, contexts_negatives, mask, label, feature):
        center_inputs = {"index": center.long(), "feature": feature.float()}
        context_inputs = {"index": contexts_negatives.long(), "feature": feature.float()}
        return (center_inputs, context_inputs), label.float(), mask.float()

    def __call__(self, batch_5tuple):
        return self._convert(*batch_5tuple)

    def __iter__(self):
        if self.loader is None:
            raise ValueError("LegacyBatchAdapter was constructed without a `loader` to iterate over")
        for batch_5tuple in self.loader:
            yield self._convert(*batch_5tuple)

    def __len__(self):
        return len(self.loader)
