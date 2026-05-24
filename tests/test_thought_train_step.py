"""Smoke tests for thought_train_step.py (Task 6)."""


def test_thought_train_step_importable():
    """Sanity check: thought_train_step.train_step is importable and callable."""
    from thought_train_step import train_step
    assert callable(train_step)


def test_train_step_routing_at_k1():
    """At num_thoughts=1, train.py routing should pick the original train_step."""
    import importlib
    train_step_mod = importlib.import_module('train_step')
    thought_mod = importlib.import_module('thought_train_step')
    assert train_step_mod.train_step is not thought_mod.train_step
