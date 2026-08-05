"""Typed config construction must tolerate inherited, strategy-specific params."""

from __future__ import annotations

import logging

import pytest

from src.components.factory import create_typed_config
from src.velocity.config import ConstrictionVelocityConfig, StandardVelocityConfig


def test_params_for_another_strategy_do_not_crash_construction():
    """A variant overriding the strategy still inherits common `*_params` keys."""
    config = create_typed_config(
        "velocity_strategy",
        {"type": "constriction", "c1": 1.49445, "c2": 1.49445, "phi1": 2.05, "phi2": 2.05},
    )

    assert isinstance(config, ConstrictionVelocityConfig)
    assert config.phi1 == pytest.approx(2.05)
    assert config.phi2 == pytest.approx(2.05)


def test_dropped_params_are_reported(caplog: pytest.LogCaptureFixture):
    """Silently discarding a tuned hyperparameter would invalidate a comparison."""
    with caplog.at_level(logging.WARNING):
        create_typed_config("velocity_strategy", {"type": "constriction", "c1": 1.0, "c2": 2.0})

    assert "c1" in caplog.text
    assert "c2" in caplog.text
    assert "constriction" in caplog.text


def test_declared_params_are_still_applied():
    config = create_typed_config("velocity_strategy", {"type": "standard", "c1": 1.1, "c2": 2.2})

    assert isinstance(config, StandardVelocityConfig)
    assert config.c1 == pytest.approx(1.1)
    assert config.c2 == pytest.approx(2.2)


def test_no_warning_when_every_param_is_accepted(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        create_typed_config("velocity_strategy", {"type": "standard", "c1": 1.1})

    assert caplog.text == ""


def test_unregistered_component_still_returns_none():
    assert create_typed_config("influence_strategy", {"type": "single_best"}) is None
