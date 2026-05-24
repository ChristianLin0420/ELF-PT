FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install setuptools==69.5.1

# PyTorch CPU-only. Transformers imports torch at top-level even though we
# only use HF tokenizers in this codebase; the GPT-2-Large PPL evaluator is
# the only torch-GPU path, and it's optional (online_eval=false).
# CPU torch avoids cuDNN/CUDA conflicts with jax[cuda12].
RUN pip install torch==2.3.0+cpu torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# JAX with CUDA12. 0.4.38 ships cuDNN/CUDA runtime as pip packages, so no
# system CUDA toolkit is required inside the container.
RUN pip install "jax[cuda12]==0.4.38" "jaxlib==0.4.38"

# Remaining deps from requirements.txt (jax[tpu] dropped; tensorflow dropped — not imported by the code).
# Pin optax/orbax to versions compatible with jax 0.4.38 so jax doesn't get bumped.
RUN pip install \
    flax==0.10.2 \
    orbax-checkpoint==0.5.23 \
    optax==0.2.5 \
    "absl-py>=1.4.0" \
    "numpy>=1.26.4,<2.0.0" \
    "ml-dtypes>=0.4.0" \
    "scipy>=1.12.0" \
    "PyYAML>=6.0.1" \
    "matplotlib>=3.8.0" \
    "requests>=2.31.0" \
    "tqdm>=4.66.0" \
    "pillow>=9.5.0" \
    "gcsfs>=2024.1.0" \
    "datasets>=2.19.0" \
    "huggingface-hub>=0.23.0" \
    "transformers>=4.41.2,<4.45.0" \
    "wandb==0.16.6" \
    "einops>=0.7.0" \
    "sacrebleu>=2.4.0" \
    "rouge-score>=0.1.2" \
    "pytest>=8.0.0"

# Re-pin JAX after all the other installs in case any dep tried to upgrade it.
RUN pip install --force-reinstall --no-deps "jax==0.4.38" "jaxlib==0.4.38"

WORKDIR /workspace/src
ENV PYTHONPATH=/workspace/src
