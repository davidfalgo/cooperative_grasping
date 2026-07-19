# Description: Standalone (non-pytest) test for general/examples/toy.py: asserts
# the acceptance criteria (learning happened, Hit@1/PR-AUC/Spearman bars, resume
# round-trip, wall-clock budget) hold for seeds 0, 1, 2.

import os
import sys

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from conditional_embedding_model.general.examples.toy import run_toy

# run_all.py --fast convention: function names in this set are skipped under
# --fast (here: the multi-seed toy run, ~60s total).
SLOW = {"test_toy_seeds"}


def test_toy_seeds():
    for seed in (0, 1, 2):
        result = run_toy(seed=seed, epochs=30, verbose=False)

        assert result["final_loss"] < 0.5 * result["first_epoch_loss"], (
            f"[seed={seed}] final_loss {result['final_loss']} not < half of "
            f"first_epoch_loss {result['first_epoch_loss']}")
        assert result["hit_at_1"] >= 0.60, f"[seed={seed}] hit_at_1 {result['hit_at_1']} < 0.60"
        assert result["pr_auc"] >= 0.80, f"[seed={seed}] pr_auc {result['pr_auc']} < 0.80"
        assert result["spearman"] > 0.5, f"[seed={seed}] spearman {result['spearman']} <= 0.5"
        assert result["resume_ok"] is True, f"[seed={seed}] resume round-trip failed"
        assert result["elapsed_seconds"] < 60, (
            f"[seed={seed}] elapsed_seconds {result['elapsed_seconds']} >= 60")

        print(f"[seed={seed}] PASS {result}")


if __name__ == "__main__":
    test_toy_seeds()
    print("\nAll toy-example tests passed.")
