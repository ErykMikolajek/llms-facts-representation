#!/usr/bin/env python3
"""Automate SAE analyses and the Gemma MoE benchmark on Kaggle.

The pipeline downloads the checkpoint emitted by the previous Kaggle kernel,
publishes it as part of a complete new version of the configured dataset, and
pushes the notebook under the selected Kaggle account.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
from typing import Any, Iterable


CONFIG_FILENAME = "config.toml"
DATASET_METADATA_FILENAME = "dataset-metadata.json"
KERNEL_METADATA_FILENAME = "kernel-metadata.json"
PLACEHOLDER_PREFIX = "UZUPELNIJ_"
TERMINAL_KERNEL_FAILURES = ("error", "failed", "failure", "cancel", "invalid")
MINIMUM_KAGGLE_VERSION = (2, 2, 4)


class PipelineError(RuntimeError):
    """Expected workflow error with a concise user-facing message."""


@dataclass(frozen=True)
class Account:
    name: str
    username: str
    config_dir: Path


@dataclass(frozen=True)
class Workflow:
    name: str
    dataset: str
    code_dataset: str
    dataset_sources: tuple[str, ...]
    model_sources: tuple[str, ...]
    checkpoint_filename: str
    checkpoint_output_relative_path: str
    notebook: Path
    work_dir: Path
    state_file: Path
    publisher_account: str
    kernel_slug: str
    kernel_title: str
    accelerator: str
    enable_internet: bool
    kernel_private: bool
    poll_seconds: int
    dataset_timeout_seconds: int
    kernel_timeout_seconds: int

    def kernel_ref(self, account: Account) -> str:
        return f"{account.username}/{self.kernel_slug}"


@dataclass(frozen=True)
class Config:
    path: Path
    kaggle_executable: str
    work_dir: Path
    default_workflow: str
    workflows: dict[str, Workflow]
    accounts: dict[str, Account]


class KaggleCli:
    def __init__(self, executable: str) -> None:
        # Prefer a console script installed next to the currently running
        # Python. This keeps the pipeline inside its virtual environment even
        # when it is invoked as `.venv/bin/python ...` without activating the
        # environment (and therefore without modifying PATH).
        executable_path = Path(executable).expanduser()
        if executable_path.parent == Path("."):
            # Do not resolve this path: virtualenv Python binaries are often
            # symlinks to the base interpreter, while their sibling console
            # scripts live in the virtualenv's own bin directory.
            python_bin_dir = str(Path(sys.executable).parent)
            resolved = shutil.which(executable, path=python_bin_dir)
            if resolved is None:
                resolved = shutil.which(executable)
        else:
            resolved = shutil.which(str(executable_path))
        if resolved is None:
            raise PipelineError(
                f"Nie znaleziono programu Kaggle CLI: {executable!r}. "
                "Zainstaluj aktualny pakiet `kaggle` i ponów próbę."
            )
        self.executable = resolved

    def check_version(self, account: Account) -> None:
        output = self.run(account, "--version", capture=True)
        match = re.search(r"\b(\d+)\.(\d+)\.(\d+)(?:\.\d+)?\b", output)
        if match is None:
            raise PipelineError(
                "Nie udało się rozpoznać wersji Kaggle CLI z outputu: "
                f"{output.strip()!r}"
            )
        version = tuple(int(part) for part in match.groups())
        if version < MINIMUM_KAGGLE_VERSION:
            minimum = ".".join(map(str, MINIMUM_KAGGLE_VERSION))
            found = ".".join(map(str, version))
            raise PipelineError(
                f"Kaggle CLI {found} jest za stare; wymagane jest co najmniej "
                f"{minimum}."
            )

    def run(
        self,
        account: Account,
        *arguments: str,
        capture: bool = False,
    ) -> str:
        command = [self.executable, *map(str, arguments)]
        print(f"[{account.name}] {shlex.join(command)}", flush=True)
        environment = os.environ.copy()
        # Explicit profiles must not be shadowed by credentials inherited from
        # the parent shell.
        for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
            environment.pop(name, None)
        environment["KAGGLE_CONFIG_DIR"] = str(account.config_dir)
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
            check=False,
        )
        output = completed.stdout or ""
        if capture and output:
            print(output, end="" if output.endswith("\n") else "\n", flush=True)
        if completed.returncode != 0:
            detail = f"\n{output.strip()}" if output.strip() else ""
            raise PipelineError(
                f"Kaggle CLI zakończył komendę kodem {completed.returncode}: "
                f"{shlex.join(command)}{detail}"
            )
        return output


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise PipelineError(f"Brak `{key}` w sekcji [{section}] konfiguracji.")
    return mapping[key]


def _relative_to_config(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_config(path: Path) -> Config:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PipelineError(
            f"Brak konfiguracji: {path}. Skopiuj lub uzupełnij plik config.toml."
        )
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    pipeline_raw = _required(raw, "pipeline", "root")
    workflows_raw = _required(raw, "workflows", "root")
    accounts_raw = _required(raw, "accounts", "root")
    work_dir = _relative_to_config(path, str(pipeline_raw.get("work_dir", ".work")))
    dataset = str(_required(pipeline_raw, "dataset", "pipeline"))
    code_dataset = str(_required(pipeline_raw, "code_dataset", "pipeline"))
    sources = tuple(str(item) for item in pipeline_raw.get("dataset_sources", []))
    if dataset not in sources:
        sources = (*sources, dataset)
    if code_dataset not in sources:
        sources = (code_dataset, *sources)

    workflows: dict[str, Workflow] = {}
    for name, workflow_raw in workflows_raw.items():
        section = f"workflows.{name}"
        workflow = Workflow(
            name=name,
            dataset=str(workflow_raw.get("dataset", dataset)),
            code_dataset=str(workflow_raw.get("code_dataset", code_dataset)),
            dataset_sources=tuple(
                str(item) for item in workflow_raw.get("dataset_sources", sources)
            ),
            model_sources=tuple(
                str(item)
                for item in workflow_raw.get(
                    "model_sources", pipeline_raw.get("model_sources", [])
                )
            ),
            checkpoint_filename=str(
                _required(workflow_raw, "checkpoint_filename", section)
            ),
            checkpoint_output_relative_path=str(
                _required(workflow_raw, "checkpoint_output_relative_path", section)
            ),
            notebook=_relative_to_config(
                path, str(_required(workflow_raw, "notebook", section))
            ),
            work_dir=work_dir,
            state_file=_relative_to_config(
                path,
                str(workflow_raw.get("state_file", f".state/{name}.json")),
            ),
            publisher_account=str(
                workflow_raw.get(
                    "publisher_account",
                    _required(pipeline_raw, "publisher_account", "pipeline"),
                )
            ),
            kernel_slug=str(_required(workflow_raw, "kernel_slug", section)),
            kernel_title=str(_required(workflow_raw, "kernel_title", section)),
            accelerator=str(
                workflow_raw.get(
                    "accelerator", pipeline_raw.get("accelerator", "NvidiaTeslaT4")
                )
            ),
            enable_internet=bool(
                workflow_raw.get(
                    "enable_internet", pipeline_raw.get("enable_internet", True)
                )
            ),
            kernel_private=bool(
                workflow_raw.get(
                    "kernel_private", pipeline_raw.get("kernel_private", True)
                )
            ),
            poll_seconds=int(
                workflow_raw.get("poll_seconds", pipeline_raw.get("poll_seconds", 60))
            ),
            dataset_timeout_seconds=int(
                workflow_raw.get(
                    "dataset_timeout_seconds",
                    pipeline_raw.get("dataset_timeout_seconds", 1800),
                )
            ),
            kernel_timeout_seconds=int(
                workflow_raw.get(
                    "kernel_timeout_seconds",
                    pipeline_raw.get("kernel_timeout_seconds", 46800),
                )
            ),
        )
        workflow_sources = list(workflow.dataset_sources)
        for required_source in (workflow.code_dataset, workflow.dataset):
            if required_source not in workflow_sources:
                workflow_sources.insert(0, required_source)
        workflow = replace(workflow, dataset_sources=tuple(workflow_sources))
        if workflow.poll_seconds < 1:
            raise PipelineError(f"{section}.poll_seconds musi być dodatnie.")
        if workflow.dataset_timeout_seconds < 1 or workflow.kernel_timeout_seconds < 1:
            raise PipelineError(f"Timeouty w [{section}] muszą być dodatnie.")
        workflows[name] = workflow
    if not workflows:
        raise PipelineError("Konfiguracja musi zawierać co najmniej jeden workflow.")

    accounts: dict[str, Account] = {}
    for name, account_raw in accounts_raw.items():
        username = str(_required(account_raw, "username", f"accounts.{name}"))
        config_dir = _relative_to_config(
            path, str(_required(account_raw, "config_dir", f"accounts.{name}"))
        )
        accounts[name] = Account(
            name=name,
            username=username,
            config_dir=config_dir,
        )
    for workflow in workflows.values():
        if workflow.publisher_account not in accounts:
            raise PipelineError(
                f"{workflow.name}.publisher_account wskazuje nieistniejące konto: "
                f"{workflow.publisher_account}"
            )
    default_workflow = str(pipeline_raw.get("default_workflow", "pythia"))
    if default_workflow not in workflows:
        raise PipelineError(
            f"Nieznany pipeline.default_workflow: {default_workflow!r}."
        )
    return Config(
        path=path,
        kaggle_executable=str(raw.get("kaggle_executable", "kaggle")),
        work_dir=work_dir,
        default_workflow=default_workflow,
        workflows=workflows,
        accounts=accounts,
    )


def account_configuration_error(account: Account) -> str | None:
    if account.username.startswith(PLACEHOLDER_PREFIX):
        return f"uzupełnij username dla [{account.name}]"
    if not account.config_dir.is_dir():
        return f"brak katalogu credentials: {account.config_dir}"
    credentials = account.config_dir / "kaggle.json"
    if not credentials.is_file():
        return f"brak pliku credentials: {credentials}"
    try:
        payload = json.loads(credentials.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return f"nie można odczytać {credentials}: {exc}"
    if not payload.get("username") or not payload.get("key"):
        return f"{credentials} nie zawiera pól username i key"
    if str(payload["username"]) != account.username:
        return (
            f"username w {credentials} ({payload['username']}) nie odpowiada "
            f"config.toml ({account.username})"
        )
    if credentials.stat().st_mode & 0o077:
        return f"niebezpieczne uprawnienia {credentials}; ustaw chmod 600"
    return None


def validate_account(account: Account) -> None:
    error = account_configuration_error(account)
    if error is not None:
        raise PipelineError(f"Niepoprawna konfiguracja konta [{account.name}]: {error}.")


def get_account(config: Config, name: str) -> Account:
    try:
        account = config.accounts[name]
    except KeyError as exc:
        available = ", ".join(sorted(config.accounts))
        raise PipelineError(
            f"Nieznane konto {name!r}. Dostępne profile: {available}."
        ) from exc
    validate_account(account)
    return account


def get_workflow(config: Config, name: str | None) -> Workflow:
    selected = name or config.default_workflow
    try:
        return config.workflows[selected]
    except KeyError as exc:
        available = ", ".join(sorted(config.workflows))
        raise PipelineError(
            f"Nieznany workflow {selected!r}. Dostępne: {available}."
        ) from exc


def read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise PipelineError(f"Nie można odczytać stanu pipeline: {path}: {exc}") from exc


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class PipelineLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self) -> "PipelineLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineError(
                f"Inna instancja pipeline już działa (blokada: {self.path})."
            ) from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(str(os.getpid()))
        self.handle.flush()
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _status_kind(output: str) -> str:
    lowered = output.lower()
    if any(word in lowered for word in TERMINAL_KERNEL_FAILURES):
        return "failed"
    if re.search(r"\bcomplete(?:d)?\b", lowered):
        return "complete"
    if re.search(r"\bready\b", lowered):
        return "ready"
    if re.search(r"\b(?:running|queued|pending)\b", lowered):
        return "pending"
    return "unknown"


def wait_for_kernel(
    cli: KaggleCli,
    account: Account,
    kernel_ref: str,
    workflow: Workflow,
    *,
    wait: bool,
    allow_failed: bool = False,
) -> None:
    deadline = time.monotonic() + workflow.kernel_timeout_seconds
    while True:
        output = cli.run(account, "kernels", "status", kernel_ref, capture=True)
        status = _status_kind(output)
        if status == "complete":
            return
        if status == "failed":
            if allow_failed:
                print(
                    f"UWAGA: run {kernel_ref} ma status błędu; próbuję pobrać "
                    "ostatni zapisany checkpoint zgodnie z --allow-failed-source."
                )
                return
            raise PipelineError(f"Poprzedni run {kernel_ref} zakończył się błędem.")
        if not wait:
            raise PipelineError(
                f"Run {kernel_ref} nie jest jeszcze zakończony. Użyj --wait-source, "
                "aby skrypt zaczekał."
            )
        if time.monotonic() >= deadline:
            raise PipelineError(f"Przekroczono timeout oczekiwania na run {kernel_ref}.")
        time.sleep(workflow.poll_seconds)


def wait_for_dataset(
    cli: KaggleCli,
    account: Account,
    workflow: Workflow,
) -> None:
    deadline = time.monotonic() + workflow.dataset_timeout_seconds
    while True:
        output = cli.run(
            account,
            "datasets",
            "status",
            workflow.dataset,
            capture=True,
        )
        status = _status_kind(output)
        if status == "ready":
            return
        if status == "failed":
            raise PipelineError(
                f"Nowa wersja datasetu {workflow.dataset} zakończyła się błędem."
            )
        if time.monotonic() >= deadline:
            raise PipelineError(
                f"Przekroczono timeout przetwarzania datasetu {workflow.dataset}."
            )
        time.sleep(workflow.poll_seconds)


def account_for_kernel(
    config: Config,
    kernel_ref: str,
    explicit_name: str | None,
    fallback: Account,
) -> Account:
    if explicit_name is not None:
        return get_account(config, explicit_name)
    owner = kernel_ref.split("/", 1)[0]
    for account in config.accounts.values():
        if account.username == owner:
            validate_account(account)
            return account
    return fallback


def resolve_source(
    config: Config,
    workflow: Workflow,
    runner: Account,
    source_kernel: str | None,
    source_account_name: str | None,
) -> tuple[str, Account]:
    state = read_state(workflow.state_file)
    if source_kernel:
        kernel_ref = source_kernel
        remembered_account = source_account_name
    elif state.get("last_launched_kernel"):
        kernel_ref = str(state["last_launched_kernel"])
        remembered_account = source_account_name or state.get("last_runner_account")
    else:
        kernel_ref = workflow.kernel_ref(runner)
        remembered_account = source_account_name
    if kernel_ref.count("/") != 1:
        raise PipelineError(
            f"Niepoprawny identyfikator kernela {kernel_ref!r}; oczekiwano owner/slug."
        )
    source_account = account_for_kernel(
        config,
        kernel_ref,
        str(remembered_account) if remembered_account else None,
        runner,
    )
    return kernel_ref, source_account


def _safe_clear_directory(path: Path, root: Path) -> None:
    root = root.resolve()
    path = path.resolve()
    if path == root or root not in path.parents:
        raise PipelineError(f"Odmowa wyczyszczenia katalogu poza work_dir: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def download_checkpoint(
    cli: KaggleCli,
    account: Account,
    kernel_ref: str,
    workflow: Workflow,
) -> tuple[Path, str]:
    download_dir = workflow.work_dir / "kernel-output"
    _safe_clear_directory(download_dir, workflow.work_dir)
    pattern = rf".*{re.escape(workflow.checkpoint_filename)}$"
    cli.run(
        account,
        "kernels",
        "output",
        kernel_ref,
        "-p",
        str(download_dir),
        "-o",
        "--file-pattern",
        pattern,
    )
    expected = download_dir / workflow.checkpoint_output_relative_path
    if expected.is_file():
        checkpoint = expected
    else:
        matches = sorted(download_dir.rglob(workflow.checkpoint_filename))
        if len(matches) != 1:
            found = ", ".join(str(path) for path in matches) or "brak"
            raise PipelineError(
                "Nie udało się jednoznacznie znaleźć checkpointu w output kernela. "
                f"Znaleziono: {found}."
            )
        checkpoint = matches[0]
    if checkpoint.stat().st_size < 1024:
        raise PipelineError(f"Pobrany checkpoint jest podejrzanie mały: {checkpoint}")
    checksum = sha256_file(checkpoint)
    print(
        f"Checkpoint: {checkpoint} ({checkpoint.stat().st_size:,} B, "
        f"sha256={checksum[:16]}…)",
        flush=True,
    )
    return checkpoint, checksum


def _parse_dataset_file_names(output: str) -> set[str]:
    lines = [line for line in output.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        try:
            header = next(csv.reader([line]))
        except csv.Error:
            continue
        normalized = [item.strip().lower() for item in header]
        if "name" not in normalized:
            continue
        reader = csv.DictReader(lines[index:])
        names = set()
        for row in reader:
            name = row.get("name") or row.get("Name")
            if name:
                names.add(name.strip())
        return names
    raise PipelineError(
        "Nie udało się odczytać listy plików datasetu z `kaggle datasets files -v`."
    )


def remote_dataset_files(
    cli: KaggleCli,
    publisher: Account,
    workflow: Workflow,
) -> set[str]:
    output = cli.run(
        publisher,
        "datasets",
        "files",
        workflow.dataset,
        "-v",
        "--page-size",
        "200",
        capture=True,
    )
    names = _parse_dataset_file_names(output)
    if len(names) >= 200:
        raise PipelineError(
            "Dataset ma co najmniej 200 plików; bieżący walidator nie obsługuje "
            "paginacji. Przerwano, aby nie opublikować niepełnej wersji."
        )
    if any("/" in name or "\\" in name for name in names):
        raise PipelineError(
            "Dataset zawiera zagnieżdżone ścieżki. Kaggle CLI nie potrafi ich "
            "bezpiecznie zachować w tym workflow."
        )
    return names


def sync_dataset(
    cli: KaggleCli,
    publisher: Account,
    workflow: Workflow,
    *,
    refresh: bool,
) -> Path:
    stage = workflow.work_dir / "dataset"
    metadata = stage / DATASET_METADATA_FILENAME
    if refresh or not metadata.is_file():
        _safe_clear_directory(stage, workflow.work_dir)
        cli.run(
            publisher,
            "datasets",
            "download",
            workflow.dataset,
            "-p",
            str(stage),
            "--unzip",
            "-o",
        )
        cli.run(
            publisher,
            "datasets",
            "metadata",
            workflow.dataset,
            "-p",
            str(stage),
        )
    if not metadata.is_file():
        raise PipelineError(f"Brak {DATASET_METADATA_FILENAME} w stagingu: {stage}")
    try:
        metadata_payload = json.loads(metadata.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Niepoprawny {metadata}: {exc}") from exc
    metadata_id = metadata_payload.get("id")
    # Kaggle CLI 2.2.x downloads metadata in the new `{info: {...}}`
    # response schema, while `datasets version` still requires the legacy
    # top-level `id` or `id_no`. Normalize the downloaded file before using it
    # for an upload, retaining descriptive fields where available.
    if not metadata_id:
        info = metadata_payload.get("info")
        if isinstance(info, dict):
            owner = info.get("ownerUser")
            slug = info.get("datasetSlug")
            discovered_id = f"{owner}/{slug}" if owner and slug else None
            if discovered_id and discovered_id != workflow.dataset:
                raise PipelineError(
                    f"Metadata wskazuje dataset {discovered_id}, "
                    f"oczekiwano {workflow.dataset}."
                )
            for key in (
                "title",
                "subtitle",
                "description",
                "keywords",
                "licenses",
            ):
                if key in info and key not in metadata_payload:
                    metadata_payload[key] = info[key]
            metadata_payload["id"] = workflow.dataset
            metadata.write_text(
                json.dumps(metadata_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            metadata_id = workflow.dataset
    if metadata_id and metadata_id != workflow.dataset:
        raise PipelineError(
            f"Metadata wskazuje dataset {metadata_id}, oczekiwano {workflow.dataset}."
        )
    if not metadata_id:
        raise PipelineError(
            f"{metadata} nie zawiera identyfikatora datasetu ani sekcji info."
        )
    return stage


def validate_complete_dataset_stage(
    stage: Path,
    remote_names: set[str],
    checkpoint_filename: str,
) -> None:
    directories = [path.name for path in stage.iterdir() if path.is_dir()]
    if directories:
        raise PipelineError(
            "Staging datasetu zawiera katalogi, które `datasets version` pominęłoby: "
            + ", ".join(sorted(directories))
        )
    local_names = {
        path.name
        for path in stage.iterdir()
        if path.is_file() and path.name != DATASET_METADATA_FILENAME
    }
    missing = remote_names - local_names
    unexpected = local_names - remote_names - {checkpoint_filename}
    if missing or unexpected:
        details = []
        if missing:
            details.append("brak lokalnie: " + ", ".join(sorted(missing)))
        if unexpected:
            details.append("nadmiar lokalnie: " + ", ".join(sorted(unexpected)))
        raise PipelineError(
            "Staging nie odpowiada aktualnej wersji datasetu ("
            + "; ".join(details)
            + "). Uruchom ponownie z --refresh-dataset."
        )


def publish_checkpoint(
    cli: KaggleCli,
    publisher: Account,
    workflow: Workflow,
    checkpoint: Path,
    checksum: str,
    source_kernel: str,
    *,
    refresh_dataset: bool,
) -> None:
    stage = sync_dataset(
        cli,
        publisher,
        workflow,
        refresh=refresh_dataset,
    )
    remote_names = remote_dataset_files(cli, publisher, workflow)
    destination = stage / workflow.checkpoint_filename
    shutil.copy2(checkpoint, destination)
    validate_complete_dataset_stage(
        stage,
        remote_names,
        workflow.checkpoint_filename,
    )
    message = (
        f"{workflow.name} analysis checkpoint from {source_kernel}; "
        f"sha256={checksum[:16]}"
    )
    cli.run(
        publisher,
        "datasets",
        "version",
        "-p",
        str(stage),
        "-m",
        message,
        "-t",
    )
    wait_for_dataset(cli, publisher, workflow)


def prepare_kernel_stage(account: Account, workflow: Workflow) -> Path:
    if not workflow.notebook.is_file():
        raise PipelineError(f"Brak notebooka: {workflow.notebook}")
    stage = workflow.work_dir / "kernels" / workflow.name / account.name
    _safe_clear_directory(stage, workflow.work_dir)
    notebook_name = workflow.notebook.name
    shutil.copy2(workflow.notebook, stage / notebook_name)
    metadata = {
        "id": workflow.kernel_ref(account),
        "title": workflow.kernel_title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": workflow.kernel_private,
        "enable_gpu": True,
        "enable_internet": workflow.enable_internet,
        "machine_shape": workflow.accelerator,
        "dataset_sources": list(workflow.dataset_sources),
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": list(workflow.model_sources),
    }
    (stage / KERNEL_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return stage


def validate_model_sources(
    cli: KaggleCli,
    account: Account,
    workflow: Workflow,
) -> None:
    """Fail before a kernel push when an account cannot use a gated model."""
    for source in workflow.model_sources:
        try:
            cli.run(
                account,
                "models",
                "instances",
                "versions",
                "files",
                source,
                "--page-size",
                "1",
                capture=True,
            )
        except PipelineError as exc:
            raise PipelineError(
                f"Konto {account.name} ({account.username}) nie ma dostępu do "
                f"Kaggle Model {source}. Otwórz stronę modelu na tym koncie, "
                "zaakceptuj warunki Gemma i ponów polecenie."
            ) from exc


def launch_kernel(
    cli: KaggleCli,
    account: Account,
    workflow: Workflow,
) -> None:
    stage = prepare_kernel_stage(account, workflow)
    cli.run(
        account,
        "kernels",
        "push",
        "-p",
        str(stage),
        "--accelerator",
        workflow.accelerator,
    )


def record_launch(
    workflow: Workflow,
    runner: Account,
    *,
    source_kernel: str | None = None,
    checkpoint_sha256: str | None = None,
) -> None:
    state = read_state(workflow.state_file)
    state.update(
        {
            "last_launched_kernel": workflow.kernel_ref(runner),
            "last_runner_account": runner.name,
            "last_launched_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if source_kernel is not None:
        state["previous_kernel"] = source_kernel
    if checkpoint_sha256 is not None:
        state["checkpoint_sha256"] = checkpoint_sha256
    write_state(workflow.state_file, state)


def command_cycle(config: Config, cli: KaggleCli, args: argparse.Namespace) -> None:
    workflow = get_workflow(config, args.workflow)
    runner = get_account(config, args.account)
    validate_model_sources(cli, runner, workflow)
    publisher = get_account(config, workflow.publisher_account)
    source_kernel, source_account = resolve_source(
        config,
        workflow,
        runner,
        args.source_kernel,
        args.source_account,
    )
    print(
        f"Workflow: {workflow.name}; źródło: {source_kernel} ({source_account.name}); "
        f"publikacja: {publisher.name}; kolejny run: {workflow.kernel_ref(runner)}",
        flush=True,
    )
    wait_for_kernel(
        cli,
        source_account,
        source_kernel,
        workflow,
        wait=args.wait_source,
        allow_failed=args.allow_failed_source,
    )
    checkpoint, checksum = download_checkpoint(
        cli,
        source_account,
        source_kernel,
        workflow,
    )
    publish_checkpoint(
        cli,
        publisher,
        workflow,
        checkpoint,
        checksum,
        source_kernel,
        refresh_dataset=args.refresh_dataset,
    )
    launch_kernel(cli, runner, workflow)
    record_launch(
        workflow,
        runner,
        source_kernel=source_kernel,
        checkpoint_sha256=checksum,
    )
    if args.wait:
        wait_for_kernel(
            cli,
            runner,
            workflow.kernel_ref(runner),
            workflow,
            wait=True,
        )


def command_launch(config: Config, cli: KaggleCli, args: argparse.Namespace) -> None:
    workflow = get_workflow(config, args.workflow)
    runner = get_account(config, args.account)
    validate_model_sources(cli, runner, workflow)
    launch_kernel(cli, runner, workflow)
    record_launch(workflow, runner)
    if args.wait:
        wait_for_kernel(
            cli,
            runner,
            workflow.kernel_ref(runner),
            workflow,
            wait=True,
        )


def command_sync_dataset(config: Config, cli: KaggleCli, args: argparse.Namespace) -> None:
    workflow = get_workflow(config, args.workflow)
    publisher = get_account(config, workflow.publisher_account)
    stage = sync_dataset(cli, publisher, workflow, refresh=True)
    remote_names = remote_dataset_files(cli, publisher, workflow)
    validate_complete_dataset_stage(
        stage,
        remote_names,
        workflow.checkpoint_filename,
    )
    print(f"Pełny staging datasetu jest gotowy: {stage}")


def command_accounts(config: Config, _cli: KaggleCli, _args: argparse.Namespace) -> None:
    for account in config.accounts.values():
        configuration_error = account_configuration_error(account)
        marker = "OK" if configuration_error is None else f"DO UZUPEŁNIENIA: {configuration_error}"
        publishers = [
            workflow.name
            for workflow in config.workflows.values()
            if account.name == workflow.publisher_account
        ]
        publisher = f" [publisher: {', '.join(publishers)}]" if publishers else ""
        kernel_refs = ", ".join(
            f"{name}={workflow.kernel_ref(account)}"
            for name, workflow in config.workflows.items()
        )
        print(
            f"{account.name}: {kernel_refs}; credentials={account.config_dir}; "
            f"{marker}{publisher}"
        )


def command_status(config: Config, cli: KaggleCli, args: argparse.Namespace) -> None:
    workflow = get_workflow(config, args.workflow)
    account = get_account(config, args.account)
    cli.run(account, "kernels", "status", workflow.kernel_ref(account))


def command_publish_assets(config: Config, cli: KaggleCli, args: argparse.Namespace) -> None:
    """Create or version a workflow's self-contained Kaggle Dataset."""
    workflow = get_workflow(config, args.workflow)
    account = get_account(config, args.account)
    directory = args.directory.expanduser().resolve()
    metadata_path = directory / DATASET_METADATA_FILENAME
    if not directory.is_dir():
        raise PipelineError(f"Brak katalogu assetów: {directory}")
    if not metadata_path.is_file():
        raise PipelineError(f"Brak {DATASET_METADATA_FILENAME}: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("id") != workflow.dataset:
        raise PipelineError(
            f"Dataset metadata wskazuje {metadata.get('id')!r}, "
            f"a workflow oczekuje {workflow.dataset!r}."
        )
    if args.create:
        cli.run(
            account,
            "datasets",
            "create",
            "-p",
            str(directory),
            "--dir-mode",
            "zip",
            "--keep-tabular",
        )
    else:
        cli.run(
            account,
            "datasets",
            "version",
            "-p",
            str(directory),
            "-m",
            args.message,
            "--dir-mode",
            "zip",
            "--keep-tabular",
        )


def build_parser(default_config: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automatyzacja analiz SAE i benchmarku Gemma MoE na Kaggle."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Plik konfiguracji (domyślnie: {default_config}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    accounts = subparsers.add_parser("accounts", help="Pokaż trzy profile kont.")
    accounts.set_defaults(handler=command_accounts)

    sync = subparsers.add_parser(
        "sync-dataset",
        help="Pobierz pełny dataset do bezpiecznego stagingu.",
    )
    sync.add_argument(
        "--workflow",
        help="Nazwa workflowu; domyślnie pipeline.default_workflow.",
    )
    sync.set_defaults(handler=command_sync_dataset)

    launch = subparsers.add_parser(
        "launch",
        help="Uruchom notebook z checkpointem już obecnym w datasecie.",
    )
    launch.add_argument(
        "--workflow",
        help="Nazwa workflowu; domyślnie pipeline.default_workflow.",
    )
    launch.add_argument("--account", required=True)
    launch.add_argument("--wait", action="store_true")
    launch.set_defaults(handler=command_launch)

    cycle = subparsers.add_parser(
        "cycle",
        help="Pobierz checkpoint, opublikuj dataset i uruchom kolejny run.",
    )
    cycle.add_argument(
        "--workflow",
        help="Nazwa workflowu; domyślnie pipeline.default_workflow.",
    )
    cycle.add_argument("--account", required=True, help="Konto dla kolejnego runu GPU.")
    cycle.add_argument(
        "--source-kernel",
        help="Poprzedni kernel owner/slug; domyślnie ostatni zapisany w stanie.",
    )
    cycle.add_argument(
        "--source-account",
        help="Profil credentials do prywatnego kernela źródłowego.",
    )
    cycle.add_argument("--wait-source", action="store_true")
    cycle.add_argument(
        "--allow-failed-source",
        action="store_true",
        help="Pobierz checkpoint także po timeout/error poprzedniego runu.",
    )
    cycle.add_argument("--refresh-dataset", action="store_true")
    cycle.add_argument("--wait", action="store_true", help="Czekaj też na nowy run.")
    cycle.set_defaults(handler=command_cycle)

    status = subparsers.add_parser("status", help="Pokaż status kernela konta.")
    status.add_argument(
        "--workflow",
        help="Nazwa workflowu; domyślnie pipeline.default_workflow.",
    )
    status.add_argument("--account", required=True)
    status.set_defaults(handler=command_status)

    publish = subparsers.add_parser(
        "publish-assets",
        help="Utwórz lub zaktualizuj prywatny Dataset assetów workflowu.",
    )
    publish.add_argument("--workflow", default="moe_benchmark")
    publish.add_argument("--account", required=True)
    publish.add_argument("--directory", type=Path, required=True)
    publish.add_argument(
        "--message", default="Update reproducible workflow assets"
    )
    publish.add_argument(
        "--create",
        action="store_true",
        help="Utwórz dataset; bez flagi opublikuj jego kolejną wersję.",
    )
    publish.set_defaults(handler=command_publish_assets)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    parser = build_parser(script_dir / CONFIG_FILENAME)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config = load_config(args.config)
        config.work_dir.mkdir(parents=True, exist_ok=True)
        cli = KaggleCli(config.kaggle_executable)
        lock_path = config.work_dir / "pipeline.lock"
        with PipelineLock(lock_path):
            if args.command != "accounts":
                workflow = get_workflow(config, getattr(args, "workflow", None))
                check_account_name = getattr(
                    args, "account", workflow.publisher_account
                )
                cli.check_version(get_account(config, check_account_name))
            args.handler(config, cli, args)
        return 0
    except PipelineError as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Przerwano.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
