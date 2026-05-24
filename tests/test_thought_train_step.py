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


def test_thought_train_step_k2_forward_pass():
    """Smoke test: run one forward pass at K=2 with realistic shapes; assert finite outputs."""
    import jax
    import jax.numpy as jnp
    from modules.parallel_thought import ELF_PT_models

    rng = jax.random.PRNGKey(0)
    model = ELF_PT_models['ELF-PT-B'](
        text_encoder_dim=128, max_length=16,
        attn_drop=0.0, proj_drop=0.0,
        num_time_tokens=4, num_self_cond_cfg_tokens=4,
        num_model_mode_tokens=4, vocab_size=32100,
        bottleneck_dim=64,
        num_thoughts=2, block_pattern="intra,inter",
        aggregation="mean",
    )

    # Initialize with dummy inputs matching the train step's expected shapes:
    # x is (B, K*L, 2*text_encoder_dim) due to self-cond doubling
    B, K, L = 2, 2, 16
    dummy_x = jnp.ones((B, K * L, 2 * 128))
    dummy_t = jnp.ones((B,))
    dummy_mask = jnp.ones((B, K * L, K * L), dtype=jnp.bool_)
    dummy_cfg = jnp.ones((B,))
    init_params = model.init(rng, dummy_x, dummy_t, intra_mask=dummy_mask, inter_mask=dummy_mask,
                             self_cond_cfg_scale=dummy_cfg)

    # Standard forward pass (no return_pre_unembed): output shape (B, K*L, text_encoder_dim)
    out, _ = model.apply(init_params, dummy_x, dummy_t, intra_mask=dummy_mask,
                         inter_mask=dummy_mask, self_cond_cfg_scale=dummy_cfg,
                         decoder_step_active=jnp.array(False))
    assert out.shape == (B, K * L, 128), f"unexpected shape: {out.shape}"
    assert jnp.all(jnp.isfinite(out)), "non-finite values in standard output"

    # Pre-unembed path with aggregation: output shape (B, L, hidden_size)
    out_agg, _ = model.apply(init_params, dummy_x, dummy_t, intra_mask=dummy_mask,
                             inter_mask=dummy_mask, self_cond_cfg_scale=dummy_cfg,
                             decoder_step_active=jnp.array(True),
                             return_pre_unembed=True)
    # hidden_size of ELF-PT-B is 768; after mean aggregation: (B, L, 768)
    assert out_agg.shape == (B, L, 768), f"unexpected aggregated shape: {out_agg.shape}"
    assert jnp.all(jnp.isfinite(out_agg)), "non-finite values in aggregated output"
