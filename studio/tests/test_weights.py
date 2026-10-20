"""Weights come from the Hugging Face short links; a local file or hf:// link overrides them."""
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from falcon_runtime import VARIANTS, FalconRuntime, resolve_checkpoint, variant_for, weights_source


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Fake huggingface_hub that records downloads; nothing is cached unless listed in `cached`."""
    calls, cached = [], set()
    def hf_hub_download(repo_id, filename):
        calls.append((repo_id, filename))
        path = tmp_path / filename; path.write_bytes(b"weights"); return str(path)
    def try_to_load_from_cache(repo_id, filename):
        return str(tmp_path / filename) if (repo_id, filename) in cached else None
    module = types.SimpleNamespace(hf_hub_download=hf_hub_download, try_to_load_from_cache=try_to_load_from_cache)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return types.SimpleNamespace(calls=calls, cached=cached)


def test_short_links_and_separate_file_names():
    assert (VARIANTS["small"]["repo"], VARIANTS["small"]["weights"]) == ("JusperLee/Falcon", "falcon.pt")
    assert (VARIANTS["big"]["repo"], VARIANTS["big"]["weights"]) == ("JusperLee/Falcon-Big", "falcon-big.pt")
    assert variant_for("JusperLee/Falcon-Big") == "big" and variant_for("small") == "small"
    with pytest.raises(ValueError):
        variant_for("someone/Other")


def test_default_weights_come_from_the_short_link(hub):
    assert resolve_checkpoint("big").name == "falcon-big.pt"
    assert hub.calls == [("JusperLee/Falcon-Big", "falcon-big.pt")]


def test_local_file_overrides_without_network(hub, tmp_path):
    local = tmp_path / "my.pt"
    assert resolve_checkpoint("small", local) == local and hub.calls == []


def test_hf_link_names_repo_and_file(hub):
    assert resolve_checkpoint("small", "hf://someone/Falcon-Custom/model.pt").name == "model.pt"
    assert hub.calls == [("someone/Falcon-Custom", "model.pt")]
    with pytest.raises(ValueError):
        resolve_checkpoint("small", "hf://model.pt")


def test_cache_status_for_the_ui(hub):
    assert weights_source("small") == "huggingface"
    hub.cached.add(("JusperLee/Falcon", "falcon.pt"))
    assert weights_source("small") == "cached"


def test_download_failure_suggests_a_mirror(monkeypatch):
    def fail(**_):
        raise OSError("network unreachable")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fail))
    with pytest.raises(FileNotFoundError, match="HF_ENDPOINT"):
        resolve_checkpoint("small")


def test_from_pretrained_takes_short_links(monkeypatch):
    loaded = []
    monkeypatch.setattr(FalconRuntime, "load", lambda self, device="auto": loaded.append((self.variant, device)) or self)
    FalconRuntime.from_pretrained("JusperLee/Falcon-Big", "cpu")
    FalconRuntime.from_pretrained()
    assert loaded == [("big", "cpu"), ("small", "auto")]
