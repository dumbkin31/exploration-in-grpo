"""hub_sync carries a run between sessions through a (here: fake, in-memory) Hugging Face Hub repo."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")
REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("hub_sync", REPO / "scripts" / "hub_sync.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class FakeApi:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.commits: list[list] = []
        self.exists = False

    def create_repo(self, repo_id, private=False, exist_ok=False):
        assert private is True
        self.exists = True

    def list_repo_files(self, repo_id):
        if not self.exists:
            raise RuntimeError("404 Client Error: Repository Not Found")
        return sorted(self.files)

    def create_commit(self, repo_id, operations, commit_message):
        self.commits.append(operations)
        for op in operations:
            if hasattr(op, "is_folder"):
                prefix = op.path_in_repo
                for k in [k for k in self.files if k.startswith(prefix)]:
                    del self.files[k]
            else:
                self.files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()


def _fake_download(api: FakeApi):
    def hf_hub_download(repo_id, filename, local_dir):
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(api.files[filename])
        return str(out)

    return hf_hub_download


def _save_step(run: Path, step: int) -> None:
    ck = run / "checkpoints" / f"global_step_{step}" / "actor"
    ck.mkdir(parents=True)
    (ck / "model_world_size_1_rank_0.pt").write_bytes(b"w" * step)
    (ck / "lora_train_meta.json").write_text('{"r": 64}')
    (run / "checkpoints" / "latest_checkpointed_iteration.txt").write_text(f"{step}\n")
    with (run / "metrics.jsonl").open("a") as f:
        f.write(f'{{"step": {step}}}\n')


def test_push_uploads_new_checkpoints_once_and_prunes(tmp_path: Path):
    mod, api = _load(), FakeApi()
    run = tmp_path / "runs" / "math_grpo-t4-s1"
    run.mkdir(parents=True)
    (run / "rollout_dumps").mkdir()
    (run / "rollout_dumps" / "1.jsonl").write_text("big")
    for step in (1, 2, 3):
        _save_step(run, step)
        mod.push(api, "u/r", tmp_path / "runs", "math_grpo-t4-s1", keep=2)
    steps = sorted({p.split("/")[2] for p in api.files if "/global_step_" in p})
    assert steps == ["global_step_2", "global_step_3"]
    assert api.files["math_grpo-t4-s1/checkpoints/latest_checkpointed_iteration.txt"] == b"3\n"
    assert not any("rollout_dumps" in p for p in api.files)
    again = mod.push(api, "u/r", tmp_path / "runs", "math_grpo-t4-s1", keep=2)
    assert again["uploaded_checkpoint"] is False, "a checkpoint already on the Hub is not uploaded again"


def test_pull_restores_the_newest_checkpoint_into_a_fresh_session(tmp_path: Path, monkeypatch):
    mod, api = _load(), FakeApi()
    src = tmp_path / "a" / "runs"
    run = src / "math_mixed_cuts-t4-s1"
    run.mkdir(parents=True)
    for step in (4, 5):
        _save_step(run, step)
        mod.push(api, "u/r", src, "math_mixed_cuts-t4-s1")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_download(api))
    fresh = tmp_path / "b" / "runs"
    got = mod.pull(api, "u/r", fresh, "math_mixed_cuts-t4-s1")
    new = fresh / "math_mixed_cuts-t4-s1"
    assert got["step"] == 5
    assert (new / "checkpoints/global_step_5/actor/model_world_size_1_rank_0.pt").read_bytes() == b"w" * 5
    assert not (new / "checkpoints/global_step_4").exists()
    assert (new / "checkpoints/latest_checkpointed_iteration.txt").read_text() == "5\n"
    assert (new / "metrics.jsonl").read_text().splitlines() == ['{"step": 4}', '{"step": 5}']
    older = mod.pull(api, "u/r", tmp_path / "c" / "runs", "math_mixed_cuts-t4-s1", step=4)
    assert (
        older["step"] == 4 and (tmp_path / "c/runs/math_mixed_cuts-t4-s1/checkpoints/global_step_4").is_dir()
    )


def test_pull_of_an_unknown_run_is_a_clean_first_session(tmp_path: Path):
    mod, api = _load(), FakeApi()
    assert mod.pull(api, "u/r", tmp_path, "math_grpo-t4-s1") == {
        "run": "math_grpo-t4-s1",
        "step": None,
        "files": 0,
    }
    assert mod.status(api, "u/r") == {}
