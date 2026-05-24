"""Smoke tests for thought_train_step_r.py (Task R5)."""


def test_thought_train_step_r_importable():
    from thought_train_step_r import train_step
    assert callable(train_step)


def test_train_step_r_routing():
    """thought_train_step_r is distinct from thought_train_step and train_step."""
    import importlib
    sym = importlib.import_module('thought_train_step').train_step
    r = importlib.import_module('thought_train_step_r').train_step
    base = importlib.import_module('train_step').train_step
    assert r is not sym
    assert r is not base
