from __future__ import annotations

import unittest

import numpy as np
import torch

from sae_pipeline.features_analysis import (
    ComprehensiveFeatureAnalyzer,
    FeatureExample,
    _analysis_checkpoint_config_matches,
    _analysis_ranges,
)


class DummyTokenizer:
    def decode(self, tokens, **_kwargs):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return " ".join(str(int(token)) for token in tokens)


class DummySae:
    d_sae = 2

    def eval(self):
        return self

    def __call__(self, activations):
        shape = (*activations.shape[:-1], self.d_sae)
        sparse = torch.zeros(shape, dtype=activations.dtype, device=activations.device)
        sparse[..., 0] = 1.0
        return activations, sparse


def make_analyzer(top_k_examples=10):
    return ComprehensiveFeatureAnalyzer(
        sae_model=DummySae(),
        llm_model=object(),
        tokenizer=DummyTokenizer(),
        device=torch.device("cpu"),
        top_k_examples=top_k_examples,
    )


class FeatureAnalysisTests(unittest.TestCase):
    def test_streaming_examples_keep_real_sequence_boundaries(self):
        analyzer = make_analyzer()
        sequence_tokens = torch.tensor([[10, 11, 0], [20, 21, 0]])
        sequence_mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
        analyzer.process_batch(
            activations=torch.ones((4, 1)),
            tokens=torch.tensor([10, 11, 20, 21]),
            batch_start_idx=100,
            sequence_tokens=sequence_tokens,
            sequence_mask=sequence_mask,
        )
        examples = [item[2] for item in analyzer._example_heaps[0]]
        self.assertEqual({example.sequence_id for example in examples}, {100, 101})
        self.assertLessEqual(max(example.position_in_seq for example in examples), 1)
        first_sequence_contexts = [
            example.full_context for example in examples if example.sequence_id == 100
        ]
        self.assertTrue(all("20" not in context for context in first_sequence_contexts))

    def test_legacy_context_migration_recovers_coordinates(self):
        analyzer = make_analyzer()
        legacy = FeatureExample(sequence_id=0, position_in_seq=2)
        analyzer._example_heaps[0] = [(1.0, 1, legacy)]
        tokens = np.array([[10, 11, 0], [20, 21, 0]])
        masks = np.array([[1, 1, 0], [1, 1, 0]], dtype=bool)
        migrated = analyzer.migrate_flattened_context_examples(
            tokens=tokens,
            masks=masks,
            chunk_sequences=2,
        )
        self.assertEqual(migrated, 1)
        self.assertEqual(legacy.sequence_id, 1)
        self.assertEqual(legacy.position_in_seq, 0)
        self.assertEqual(legacy.trigger_token_id, 20)
        self.assertNotIn("11", legacy.context_before)

    def test_uniform_ranges_are_disjoint_and_exact(self):
        ranges = _analysis_ranges(50, 49, 32, "uniform")
        sampled = [index for start, end in ranges for index in range(start, end)]
        self.assertEqual(len(sampled), 49)
        self.assertEqual(len(set(sampled)), 49)

    def test_checkpoint_paths_may_move_between_kaggle_mount_layouts(self):
        loaded = {
            "tokens_path": "/kaggle/input/datasets/u/d/tokens.npy",
            "attention_mask_path": "/kaggle/input/datasets/u/d/mask.npy",
            "layer_num": 6,
        }
        expected = {
            "tokens_path": "/kaggle/input/d/tokens.npy",
            "attention_mask_path": "/kaggle/input/d/mask.npy",
            "layer_num": 6,
        }
        self.assertTrue(_analysis_checkpoint_config_matches(loaded, expected))


if __name__ == "__main__":
    unittest.main()
