from __future__ import annotations

from unittest.mock import patch
import unittest

import torch

from sae_pipeline.gemma_scope_analysis import (
    _analysis_ranges,
    _checkpoint_config_matches,
    _model_dtype,
)


class GemmaScopeAnalysisTests(unittest.TestCase):
    def test_uniform_ranges_are_disjoint_and_exact(self):
        ranges = _analysis_ranges(50, 49, 32, "uniform")
        sampled = [index for start, end in ranges for index in range(start, end)]
        self.assertEqual(len(sampled), 49)
        self.assertEqual(len(set(sampled)), 49)

    def test_checkpoint_paths_may_move_between_kaggle_mount_layouts(self):
        loaded = {
            "tokens_path": "/kaggle/input/datasets/u/d/tokens.npy",
            "attention_mask_path": "/kaggle/input/datasets/u/d/mask.npy",
            "layer_num": 9,
        }
        expected = {
            "tokens_path": "/kaggle/input/d/tokens.npy",
            "attention_mask_path": "/kaggle/input/d/mask.npy",
            "layer_num": 9,
        }
        self.assertTrue(_checkpoint_config_matches(loaded, expected))

    def test_auto_dtype_uses_float16_when_gpu_lacks_bfloat16(self):
        with patch.object(torch.cuda, "is_bf16_supported", return_value=False):
            self.assertEqual(
                _model_dtype(torch.device("cuda"), "auto"),
                torch.float16,
            )


if __name__ == "__main__":
    unittest.main()
