from __future__ import annotations

import tempfile
from pathlib import Path

from conftest import load_module

config_mod = load_module("clawtrace_config", "config.py")


def test_load_config_returns_defaults_when_no_path():
    config = config_mod.load_config()
    assert config == config_mod.DEFAULT_CONFIG


def test_run_config_from_dict_filters_unknown_keys_and_round_trips():
    data = {**config_mod.DEFAULT_CONFIG, "bogus_key": "ignored", "max_steps": 2}
    cfg = config_mod.RunConfig.from_dict(data)

    assert cfg.max_steps == 2
    assert cfg.subagent_parallelism == "serial"
    round_trip = cfg.to_dict()
    assert "bogus_key" not in round_trip
    assert round_trip["max_steps"] == 2


def test_save_and_load_config_round_trip(tmp_path):
    config_path = tmp_path / "config.json"
    original = dict(config_mod.DEFAULT_CONFIG)
    original["max_steps"] = 3
    original["backend"] = "vllm"

    config_mod.save_config(original, config_path)
    loaded = config_mod.load_config(config_path)

    assert loaded["max_steps"] == 3
    assert loaded["backend"] == "vllm"
    assert loaded["model"] == config_mod.DEFAULT_CONFIG["model"]


def main() -> None:
    test_load_config_returns_defaults_when_no_path()
    test_run_config_from_dict_filters_unknown_keys_and_round_trips()
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_save_and_load_config_round_trip(Path(tmp_dir))
    print("All config tests passed.")


if __name__ == "__main__":
    main()
