from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_pipeline as pipeline


class KagglePipelineTests(unittest.TestCase):
    def test_publish_assets_uses_declared_private_dataset(self):
        class FakeCli:
            def __init__(self):
                self.calls = []

            def run(self, account, *arguments, **kwargs):
                self.calls.append((account.name, arguments))
                return ""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / pipeline.DATASET_METADATA_FILENAME
            metadata.write_text(json.dumps({"id": "owner/assets"}))
            workflow = SimpleNamespace(dataset="owner/assets")
            credentials = root / "credentials"
            credentials.mkdir()
            credential_file = credentials / "kaggle.json"
            credential_file.write_text(json.dumps({"username": "owner", "key": "test"}))
            credential_file.chmod(0o600)
            account = pipeline.Account(name="one", username="owner", config_dir=credentials)
            config = SimpleNamespace(
                default_workflow="moe_benchmark",
                workflows={"moe_benchmark": workflow},
                accounts={"one": account},
            )
            cli = FakeCli()
            args = SimpleNamespace(
                workflow="moe_benchmark",
                account="one",
                directory=root,
                create=True,
                message="v3",
            )
            pipeline.command_publish_assets(config, cli, args)
            self.assertEqual(
                cli.calls,
                [
                    (
                        "one",
                        (
                            "datasets",
                            "create",
                            "-p",
                            str(root.resolve()),
                            "--dir-mode",
                            "zip",
                            "--keep-tabular",
                        ),
                    )
                ],
            )

    def test_parse_dataset_file_names(self):
        output = (
            "name,totalBytes,creationDate\n"
            "attention_mask_pythia.npy,10,2026-01-01\n"
            "analysis_checkpoint_pythia.pt,20,2026-01-02\n"
        )
        self.assertEqual(
            pipeline._parse_dataset_file_names(output),
            {"attention_mask_pythia.npy", "analysis_checkpoint_pythia.pt"},
        )

    def test_complete_dataset_stage_rejects_missing_remote_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            (stage / pipeline.DATASET_METADATA_FILENAME).write_text("{}")
            (stage / "analysis_checkpoint_pythia.pt").write_bytes(b"checkpoint")
            with self.assertRaises(pipeline.PipelineError):
                pipeline.validate_complete_dataset_stage(
                    stage,
                    {"analysis_checkpoint_pythia.pt", "model.pt"},
                    "analysis_checkpoint_pythia.pt",
                )

    def test_load_config_resolves_paths_relative_to_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pythia.ipynb").write_text("{}")
            (root / "gemma.ipynb").write_text("{}")
            credentials = root / "credentials"
            credentials.mkdir()
            config_path = root / "config.toml"
            config_path.write_text(
                """
[pipeline]
default_workflow = "pythia"
dataset = "owner/data"
code_dataset = "owner/code"
publisher_account = "one"

[workflows.pythia]
checkpoint_filename = "checkpoint.pt"
checkpoint_output_relative_path = "output/checkpoint.pt"
notebook = "pythia.ipynb"
kernel_slug = "pythia-analysis"
kernel_title = "Pythia analysis"

[workflows.gemma]
checkpoint_filename = "gemma-checkpoint.pt"
checkpoint_output_relative_path = "gemma/output/checkpoint.pt"
notebook = "gemma.ipynb"
kernel_slug = "gemma-analysis"
kernel_title = "Gemma analysis"
model_sources = ["google/gemma/transformers/gemma-test/1"]

[accounts.one]
username = "owner"
config_dir = "credentials"
""".strip()
            )
            config = pipeline.load_config(config_path)
            self.assertEqual(
                config.workflows["pythia"].notebook,
                (root / "pythia.ipynb").resolve(),
            )
            self.assertEqual(config.accounts["one"].config_dir, credentials.resolve())
            self.assertEqual(set(config.workflows), {"pythia", "gemma"})
            self.assertEqual(
                config.workflows["gemma"].kernel_ref(config.accounts["one"]),
                "owner/gemma-analysis",
            )
            self.assertIn("owner/data", config.workflows["pythia"].dataset_sources)
            self.assertIn("owner/code", config.workflows["gemma"].dataset_sources)
            self.assertEqual(
                config.workflows["gemma"].model_sources,
                ("google/gemma/transformers/gemma-test/1",),
            )
            self.assertEqual(config.workflows["pythia"].model_sources, ())
            stage = pipeline.prepare_kernel_stage(
                config.accounts["one"], config.workflows["gemma"]
            )
            metadata = json.loads(
                (stage / pipeline.KERNEL_METADATA_FILENAME).read_text()
            )
            self.assertEqual(
                metadata["model_sources"],
                ["google/gemma/transformers/gemma-test/1"],
            )
            dataset_stage = config.workflows["gemma"].work_dir / "dataset"
            dataset_stage.mkdir(parents=True)
            dataset_metadata = dataset_stage / pipeline.DATASET_METADATA_FILENAME
            dataset_metadata.write_text(
                json.dumps(
                    {
                        "info": {
                            "ownerUser": "owner",
                            "datasetSlug": "data",
                            "title": "Data",
                        }
                    }
                )
            )
            pipeline.sync_dataset(
                None,
                config.accounts["one"],
                config.workflows["gemma"],
                refresh=False,
            )
            normalized = json.loads(dataset_metadata.read_text())
            self.assertEqual(normalized["id"], "owner/data")
            self.assertEqual(normalized["title"], "Data")

    def test_state_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            expected = {"last_launched_kernel": "owner/kernel"}
            pipeline.write_state(path, expected)
            self.assertEqual(pipeline.read_state(path), expected)
            json.loads(path.read_text())


if __name__ == "__main__":
    unittest.main()
