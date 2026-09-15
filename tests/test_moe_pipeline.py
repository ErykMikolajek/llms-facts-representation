import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from domain_mapping.domain_mlp_activation_mapping import (
    describe_mlp,
    get_mlp_module,
    register_mlp_post_activation_hook,
    save_pooled_domain_mlp_activations,
    set_mlp_module,
    validate_sae_resid_post_compatibility,
)
from domain_mapping.domain_mlp_pruning import (
    apply_mlp_mask,
    estimate_per_domain_null_tau,
    load_domain_arrays,
)
from moe.moe_assembly import (
    CompactPrunedMLP,
    HardRoutedMLP,
    build_single_mask_expert_model,
    load_mask_expert_bank,
)
from evaluation.moe_validation import make_matched_control_masks, validate_evaluation_independence
from moe.router_training import (
    calibrate_confidence_threshold,
    LinearDomainRouter,
    RouterSample,
    load_router_samples,
    split_samples_by_domain,
)
from domain_triage.semantic_domain_triage import (
    DomainInfo,
    build_validation_sets_from_feature_analysis,
    load_precomputed_feature_analysis,
    validate_feature_analysis_contract,
)
from domain_mapping.topographic_mlp_sae_mapping import pearson_by_neuron
from domain_mapping.topographic_mlp_sae_mapping import DomainSpec
from domain_mapping.validate_domain_sae_selectivity import (
    aggregate_token_signals_by_sample,
    evaluate_selectivity,
    standardized_mean_difference,
)


class DummyGatedMLP(nn.Module):
    def __init__(self, d_model=2, d_inner=4):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)
        self.up_proj = nn.Linear(d_model, d_inner, bias=False)
        self.down_proj = nn.Linear(d_inner, d_model, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DummyModel(nn.Module):
    def __init__(self, mlp=None):
        super().__init__()
        layer = nn.Module()
        layer.mlp = mlp or DummyGatedMLP()
        backbone = nn.Module()
        backbone.layers = nn.ModuleList([layer])
        self.model = backbone
        self.config = SimpleNamespace(hidden_size=2)


class ScaleExpert(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class FakeTokenizer:
    all_special_ids = [0]
    vocab_size = 5

    def __len__(self):
        return 5

    def decode(self, ids):
        return {0: "<pad>", 1: " alpha", 2: " beta", 3: "gamma", 4: " delta"}[int(ids[0])]

    def encode(self, token, add_special_tokens=False):
        del add_special_tokens
        inverse = {self.decode([idx]): idx for idx in range(5)}
        return [inverse[token]] if token in inverse else []

    def convert_ids_to_tokens(self, token_id):
        return {0: "<pad>", 1: "▁alpha", 2: "▁beta", 3: "gamma", 4: "▁delta"}[int(token_id)]

    def get_vocab(self):
        return {self.convert_ids_to_tokens(idx): idx for idx in range(5)}


class MoePipelineTests(unittest.TestCase):
    def test_matched_controls_preserve_selected_expert_width(self):
        mlp = DummyGatedMLP(d_model=3, d_inner=6)
        selected = np.asarray([True, False, True, False, False, True])
        controls = make_matched_control_masks(mlp, selected, seed=4)
        self.assertEqual(set(controls), {"random_matched", "magnitude_matched"})
        self.assertTrue(all(int(mask.sum()) == 3 for mask in controls.values()))

    def test_router_threshold_calibration_limits_general_routing(self):
        router = LinearDomainRouter(1, 2)
        with torch.no_grad():
            router.linear.weight.copy_(torch.tensor([[1.0], [-1.0]]))
            router.linear.bias.zero_()
        general = torch.tensor([[0.0], [0.1], [0.2], [0.3], [1.0]], dtype=torch.float16)
        domain = {
            "features": torch.tensor([[2.0], [-2.0]], dtype=torch.float16),
            "labels": torch.tensor([0, 1]),
        }
        result = calibrate_confidence_threshold(
            router, general, domain, target_general_route_rate=0.2,
            device=torch.device("cpu"), batch_size=2
        )
        self.assertLessEqual(result["general_token_route_rate"], 0.2)
        self.assertEqual(result["domain_routed_accuracy"], 1.0)

    def test_selectivity_aggregates_tokens_without_document_length_weighting(self):
        sample_ids = np.asarray([10, 10, 20, 20, 20])
        source_domains = np.asarray([0, 0, 1, 1, 1])
        signals = np.asarray([[1.0, 4.0], [3.0, 2.0], [2.0, 8.0], [4.0, 4.0], [6.0, 6.0]])
        unique, labels, means = aggregate_token_signals_by_sample(
            sample_ids, source_domains, signals
        )
        np.testing.assert_array_equal(unique, [10, 20])
        np.testing.assert_array_equal(labels, [0, 1])
        np.testing.assert_allclose(means, [[2.0, 3.0], [4.0, 6.0]])

    def test_selectivity_gate_accepts_separated_domain_signals(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        signals = np.asarray(
            [[3.0, 0.0], [4.0, 1.0], [5.0, 0.5], [0.0, 3.0], [1.0, 4.0], [0.5, 5.0]]
        )
        rows, pairwise = evaluate_selectivity(
            labels, signals, [0, 1], ["a", "b"], n_bootstrap=20, seed=1, minimum_auc=0.6
        )
        self.assertTrue(all(row["passes_prespecified_gate"] for row in rows))
        self.assertEqual(len(pairwise), 2)

    def test_standardized_difference_stays_finite_for_constant_groups(self):
        effect = standardized_mean_difference(np.ones(3), np.zeros(4))
        self.assertTrue(np.isfinite(effect))
        self.assertGreater(effect, 0.0)
        self.assertEqual(standardized_mean_difference(np.ones(3), np.ones(4)), 0.0)

    def test_gemma_style_mlp_hook_and_mask_cover_both_input_projections(self):
        mlp = DummyGatedMLP()
        spec = describe_mlp(mlp)
        self.assertEqual(spec["family"], "gated_mlp")
        capture, handle = register_mlp_post_activation_hook(mlp)
        try:
            mlp(torch.ones(2, 2))
        finally:
            handle.remove()
        self.assertEqual(tuple(capture["activations"].shape), (2, 4))

        keep = np.asarray([True, False, True, False])
        stats = apply_mlp_mask(mlp, keep, zero_bias=True)
        self.assertEqual(stats["n_kept"], 2)
        self.assertTrue(torch.count_nonzero(mlp.gate_proj.weight[[1, 3]]) == 0)
        self.assertTrue(torch.count_nonzero(mlp.up_proj.weight[[1, 3]]) == 0)
        self.assertTrue(torch.count_nonzero(mlp.down_proj.weight[:, [1, 3]]) == 0)

    def test_compact_gated_expert_matches_masked_dense_mlp(self):
        torch.manual_seed(11)
        dense = DummyGatedMLP(d_model=3, d_inner=6)
        keep = torch.tensor([True, False, True, False, False, True])
        apply_mlp_mask(dense, keep.numpy(), zero_bias=True)
        compact = CompactPrunedMLP(dense, keep)
        inputs = torch.randn(2, 4, 3)

        torch.testing.assert_close(compact(inputs), dense(inputs))
        self.assertEqual(compact.n_kept, 3)
        self.assertEqual(compact.input_projections["gate_proj"].out_features, 3)
        self.assertEqual(compact.input_projections["up_proj"].out_features, 3)
        self.assertEqual(compact.output_projection.in_features, 3)

    def test_mask_only_expert_bank_matches_dense_pruning(self):
        torch.manual_seed(13)
        base = DummyGatedMLP(d_model=3, d_inner=6).eval()
        keep = np.asarray([True, False, True, False, False, True])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "domain_7_mask.npy", keep)
            (root / "pruning_summary.json").write_text(
                json.dumps(
                    {
                        "model_name": "dummy",
                        "layer_num": 0,
                        "domains": [
                            {
                                "domain_id": 7,
                                "n_kept": 3,
                                "diagnostic_empty_expert": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            bank = load_mask_expert_bank(
                root,
                base,
                domain_ids=[7],
                layer_num=0,
                model_name="dummy",
                device=torch.device("cpu"),
            )
        expected = copy.deepcopy(base)
        apply_mlp_mask(expected, keep, zero_bias=True)
        inputs = torch.randn(2, 4, 3)
        torch.testing.assert_close(bank["0"](inputs), expected(inputs))

    def test_build_single_mask_expert_model_replaces_dense_mlp(self):
        torch.manual_seed(17)
        dense = DummyGatedMLP(d_model=3, d_inner=6).eval()
        expected = copy.deepcopy(dense)
        keep = np.asarray([True, False, True, False, False, True])
        apply_mlp_mask(expected, keep, zero_bias=True)
        model = DummyModel(mlp=dense)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "domain_2_mask.npy", keep)
            (root / "pruning_summary.json").write_text(
                json.dumps(
                    {
                        "model_name": "dummy",
                        "layer_num": 0,
                        "domains": [{
                            "domain_id": 2,
                            "n_kept": 3,
                            "diagnostic_empty_expert": False,
                        }],
                    }
                ),
                encoding="utf-8",
            )
            build_single_mask_expert_model(
                model,
                experts_dir=root,
                domain_id=2,
                layer_num=0,
                model_name="dummy",
            )
        inputs = torch.randn(2, 4, 3)
        self.assertIsInstance(get_mlp_module(model, 0), CompactPrunedMLP)
        torch.testing.assert_close(get_mlp_module(model, 0)(inputs), expected(inputs))

    def test_real_transformers_gemma3_mlp_adapter_when_available(self):
        try:
            from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
            from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP
        except ImportError:
            self.skipTest("Installed Transformers does not include Gemma 3")

        config = Gemma3TextConfig(
            hidden_size=8,
            intermediate_size=16,
            hidden_activation="gelu_pytorch_tanh",
        )
        dense = Gemma3MLP(config).eval()
        inputs = torch.randn(2, 3, 8)
        keep = torch.tensor([(idx % 2) == 0 for idx in range(16)])
        capture, handle = register_mlp_post_activation_hook(dense)
        try:
            dense(inputs)
        finally:
            handle.remove()
        self.assertEqual(tuple(capture["activations"].shape), (2, 3, 16))

        apply_mlp_mask(dense, keep.numpy(), zero_bias=True)
        compact = CompactPrunedMLP(dense, keep).eval()
        torch.testing.assert_close(compact(inputs), dense(inputs))

    def test_generic_layer_adapter_supports_model_layers(self):
        model = DummyModel()
        original = get_mlp_module(model, 0)
        replacement = DummyGatedMLP()
        set_mlp_module(model, 0, replacement)
        self.assertIs(get_mlp_module(model, 0), replacement)
        self.assertIsNot(original, replacement)

    def test_sae_contract_rejects_wrong_layer(self):
        metadata = SimpleNamespace(hook_name="blocks.1.hook_resid_post")
        sae = SimpleNamespace(cfg=SimpleNamespace(d_in=2, d_sae=4, metadata=metadata))
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_sae_resid_post_compatibility(sae, DummyModel(), layer_num=0)

    def test_hard_router_uses_selected_tokens_and_dense_fallback(self):
        router = LinearDomainRouter(2, 2)
        with torch.no_grad():
            router.linear.weight.copy_(torch.tensor([[10.0, 0.0], [-10.0, 0.0]]))
            router.linear.bias.zero_()
        moe = HardRoutedMLP(
            router=router,
            experts=nn.ModuleDict({"0": ScaleExpert(2.0), "1": ScaleExpert(-2.0)}),
            domain_ids=[10, 20],
            domain_names=["a", "b"],
            fallback_expert=ScaleExpert(3.0),
            confidence_threshold=0.6,
        )
        inputs = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]])
        outputs = moe(inputs)
        expected = torch.tensor([[[2.0, 0.0], [2.0, 0.0], [0.0, 3.0]]])
        torch.testing.assert_close(outputs, expected)
        stats = moe.routing_stats()
        self.assertEqual(stats["fallback_tokens"], 1)
        self.assertEqual([row["tokens"] for row in stats["experts"]], [1, 1])

    def test_hard_router_telemetry_excludes_padding_tokens(self):
        router = LinearDomainRouter(2, 2)
        with torch.no_grad():
            router.linear.weight.copy_(torch.tensor([[10.0, 0.0], [-10.0, 0.0]]))
            router.linear.bias.zero_()
        moe = HardRoutedMLP(
            router=router,
            experts=nn.ModuleDict({"0": ScaleExpert(1.0), "1": ScaleExpert(1.0)}),
            domain_ids=[0, 1],
            domain_names=["a", "b"],
            fallback_expert=ScaleExpert(1.0),
            confidence_threshold=0.6,
        )
        moe.set_routing_token_mask(torch.tensor([[1, 1, 0]], dtype=torch.long))
        moe(torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]]))
        stats = moe.routing_stats()
        self.assertEqual(stats["total_tokens"], 2)
        self.assertEqual([row["tokens"] for row in stats["experts"]], [1, 1])
        self.assertEqual(moe.last_expert_ids.tolist(), [[0, 1, -2]])

    def test_router_split_keeps_source_groups_disjoint(self):
        samples = []
        for domain_id in (0, 1):
            for group in range(4):
                for item in range(2):
                    samples.append(
                        RouterSample(
                            text=f"d{domain_id} g{group} item{item}",
                            domain_id=domain_id,
                            domain_name=str(domain_id),
                            class_idx=domain_id,
                            group_id=f"d{domain_id}:g{group}",
                        )
                    )
        train, validation = split_samples_by_domain(samples, val_fraction=0.25, seed=7)
        self.assertFalse({sample.group_id for sample in train} & {sample.group_id for sample in validation})

    def test_router_rejects_discovery_contexts_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            domain_dir = Path(directory)
            (domain_dir / "domains.json").write_text(
                json.dumps(
                    {
                        "domains": [
                            {"domain_id": 0, "name": "a", "feature_ids": [1]},
                            {"domain_id": 1, "name": "b", "feature_ids": [2]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            validation_dir = domain_dir / "domain_validation"
            validation_dir.mkdir()
            for domain_id in (0, 1):
                rows = [
                    {
                        "text": f"domain {domain_id} sample {idx}",
                        "diagnostic_only": True,
                        "source_group": f"d{domain_id}:{idx}",
                    }
                    for idx in range(2)
                ]
                (validation_dir / f"domain_{domain_id}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )

            with self.assertRaisesRegex(ValueError, "Skipped diagnostic contexts"):
                load_router_samples(domain_dir, None, None, min_samples_per_domain=1)
            samples, _, _ = load_router_samples(
                domain_dir,
                None,
                None,
                min_samples_per_domain=1,
                allow_diagnostic_contexts=True,
            )
            self.assertEqual(len(samples), 4)

    def test_final_evaluation_requires_independent_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            discovery = root / "discovery.csv"
            discovery.write_text("text\nexample\n", encoding="utf-8")
            (root / "domains.json").write_text(
                json.dumps({"validation_source": str(discovery)}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Non-independent"):
                validate_evaluation_independence(
                    root,
                    general_eval_csv=discovery,
                    domain_eval_csv=None,
                    allow_development_eval=False,
                )
            validate_evaluation_independence(
                root,
                general_eval_csv=discovery,
                domain_eval_csv=None,
                allow_development_eval=True,
            )

    def test_final_evaluation_rejects_domain_development_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = root / "development.csv"
            general = root / "holdout_general.csv"
            development.write_text("domain_id,text\n0,example\n", encoding="utf-8")
            general.write_text("text\ngeneral\n", encoding="utf-8")
            (root / "domains.json").write_text(
                json.dumps({"validation_source": str(development)}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "development source"):
                validate_evaluation_independence(
                    root,
                    general_eval_csv=general,
                    domain_eval_csv=development,
                    allow_development_eval=False,
                )

    def test_final_evaluation_rejects_diagnostic_csv_even_when_labeled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "domains.json").write_text(
                json.dumps({"validation_source": "skipped"}), encoding="utf-8"
            )
            general = root / "general.csv"
            domain = root / "domain.csv"
            rows = "domain_id,text,diagnostic_only\n0,example,true\n"
            general.write_text(rows, encoding="utf-8")
            domain.write_text(rows, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "diagnostic_only"):
                validate_evaluation_independence(
                    root,
                    general_eval_csv=general,
                    domain_eval_csv=domain,
                    allow_development_eval=False,
                )

    def test_chunked_pearson_matches_reference(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(101, 5)).astype(np.float32)
        y = rng.normal(size=101).astype(np.float32)
        actual = pearson_by_neuron(x, y, chunk_rows=13)
        expected = np.asarray([np.corrcoef(x[:, idx], y)[0, 1] for idx in range(x.shape[1])])
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_block_max_stat_null_is_finite_and_records_method(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(128, 8)).astype(np.float32)
        y = rng.normal(size=128).astype(np.float32)
        result = estimate_per_domain_null_tau(x, y, n_permutations=8, percentile=95, seed=3, block_size=8)
        self.assertTrue(np.isfinite(result["tau"]))
        self.assertEqual(result["null_statistic"], "max_abs_correlation_across_neurons")
        reference_rng = np.random.default_rng(3)
        blocks = [y[start : start + 8] for start in range(0, len(y), 8)]
        reference_maxima = []
        for _ in range(8):
            shuffled = np.concatenate(
                [blocks[int(index)] for index in reference_rng.permutation(len(blocks))]
            )
            reference_maxima.append(float(np.nanmax(np.abs(pearson_by_neuron(x, shuffled)))))
        self.assertAlmostEqual(
            result["tau"], float(np.percentile(reference_maxima, 95)), places=5
        )

    def test_shared_pooled_mapping_selects_the_requested_domain_column(self):
        domains = [
            DomainSpec(domain_id=2, name="a", feature_ids=[1]),
            DomainSpec(domain_id=7, name="b", feature_ids=[2]),
        ]
        mlp = np.arange(12, dtype=np.float32).reshape(3, 4)
        signals = np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
        metadata = {
            "token_ids": np.arange(3, dtype=np.int64),
            "sample_indices": np.arange(3, dtype=np.int64),
            "positions": np.arange(3, dtype=np.int64),
            "source_domain_ids": np.asarray([2, 2, 7], dtype=np.int64),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pooled.npz"
            save_pooled_domain_mlp_activations(
                path,
                domains,
                mlp,
                signals,
                metadata,
                layer_num=9,
                cluster_aggregation="sum",
                activation_source="test",
                mlp_family="gated_mlp",
                compressed=True,
            )
            loaded = load_domain_arrays(path, cluster_column=1)
            with np.load(path, allow_pickle=False) as payload:
                np.testing.assert_array_equal(
                    payload["source_domain_ids"], metadata["source_domain_ids"]
                )
        np.testing.assert_array_equal(loaded["mlp_activations"], mlp)
        np.testing.assert_array_equal(loaded["cluster_activations"], signals[:, 1])
        self.assertEqual(loaded["layer_num"], 9)

    def test_precomputed_gemma_analysis_uses_token_ids_and_declared_width(self):
        payload = {
            "summary": {
                "total_features": 4,
                "layer_num": 9,
                "site": "resid_post",
                "sae_release": "release",
                "sae_id": "sae",
            },
            "features": {
                "1": {
                    "feature_id": 1,
                    "total_activations": 2,
                    "top_promoted_tokens": [
                        {"token_id": 1, "token": " alpha", "logit": 2.0},
                        {"token_id": 3, "token": "gamma", "logit": 1.0},
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature_analysis.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            matrix, tokens, metadata = load_precomputed_feature_analysis(
                path,
                FakeTokenizer(),
                min_logit=0.0,
                keep_common_function_tokens=True,
                min_token_chars=2,
                require_token_boundary=True,
                observed_features_only=True,
                use_tfidf=False,
            )
        self.assertEqual(matrix.shape, (4, 5))
        self.assertEqual(matrix.getnnz(), 1)
        self.assertEqual(tokens[1][0]["token_id"], 1)
        self.assertEqual(metadata["included_features"], 1)

    def test_trigger_token_representation_respects_top_m_without_logit_lens(self):
        payload = {
            "summary": {
                "total_features": 3,
                "layer_num": 9,
                "site": "resid_post",
                "logit_lens": "disabled",
            },
            "features": [
                {
                    "feature_id": 1,
                    "observed_in_analysis": True,
                    "common_trigger_tokens": [
                        {"token_id": 1, "token": " alpha", "count": 9},
                        {"token_id": 2, "token": " beta", "count": 8},
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature_analysis.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            matrix, tokens, metadata = load_precomputed_feature_analysis(
                path,
                FakeTokenizer(),
                min_logit=0.0,
                keep_common_function_tokens=True,
                min_token_chars=2,
                require_token_boundary=True,
                observed_features_only=True,
                use_tfidf=False,
                top_m=1,
                feature_token_source="trigger",
            )
            validate_feature_analysis_contract(
                metadata,
                layer_num=9,
                tokenizer=FakeTokenizer(),
                requested_top_m=1,
                allow_mismatch=False,
                feature_token_source="trigger",
            )
        self.assertEqual(matrix.getnnz(), 1)
        self.assertEqual(tokens[1][0]["source"], "observed_trigger_token")
        self.assertEqual(metadata["feature_token_source"], "trigger")

    def test_context_validation_filters_by_domain_without_mutating_candidates(self):
        domain = DomainInfo(
            domain_id=0,
            cluster_label=7,
            name="alpha",
            size=1,
            cohesion=1.0,
            selection_score=1.0,
            top_tokens=[
                {
                    "token": " alpha",
                    "normalized": "alpha",
                    "score": 1.0,
                    "feature_count": 1,
                }
            ],
            feature_ids=[1],
        )
        analysis = {
            "features": [
                {
                    "feature_id": 1,
                    "top_examples": [
                        {
                            "full_context": "alpha biology context",
                            "activation_score": 2.0,
                            "sequence_id": 11,
                        },
                        {
                            "full_context": "unrelated text",
                            "activation_score": 3.0,
                            "sequence_id": 12,
                        },
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis_path = root / "analysis.json"
            analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
            summary = build_validation_sets_from_feature_analysis(
                [domain], analysis_path, root / "validation", samples_per_domain=1
            )
            row = json.loads(
                (root / "validation" / "domain_0.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
        self.assertEqual(summary["domains"][0]["available_contexts"], 1)
        self.assertIn("alpha", row["text"])
        self.assertGreater(row["domain_lexical_score"], 0.0)

    def test_feature_analysis_contract_rejects_tokenizer_fingerprint_mismatch(self):
        metadata = {
            "source_format": {
                "layer_num": 9,
                "site": "resid_post",
                "logit_lens": "approximate_resid_decoder_to_unembedding",
                "logit_top_k": 128,
                "tokenizer_vocab_size": 5,
                "tokenizer_vocab_sha256": "not-the-loaded-tokenizer",
            }
        }
        with self.assertRaisesRegex(ValueError, "different hashes"):
            validate_feature_analysis_contract(
                metadata,
                layer_num=9,
                tokenizer=FakeTokenizer(),
                requested_top_m=128,
                allow_mismatch=False,
            )


if __name__ == "__main__":
    unittest.main()
