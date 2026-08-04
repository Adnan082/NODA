# CPU image, deliberately -- matches this project's own determinism requirement for
# data generation (CLAUDE.md: GPU float reductions aren't guaranteed bit-associative,
# so trajectory generation is pinned to CPU regardless of what hardware trains on).
# Training on AWS GPU instances is a separate, already-established workflow
# (infra/aws/); this image is for reproducing the environment, running the test
# suite, and running eval/*.py against already-trained checkpoints -- not training.
FROM python:3.11-slim

WORKDIR /app

# jax-cfd and friends need a C build toolchain for some transitive deps' sdists.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

ENV JAX_PLATFORM_NAME=cpu

# Build-time sanity check: the image is only useful if the test suite actually
# passes in it. Checkpoint-dependent tests skip (no trained weights baked into the
# image -- see .dockerignore), same as a fresh git checkout.
RUN PYTHONPATH=src pytest -q

# Interactive use (make bench, make data, training against a mounted checkpoints/
# volume, etc.) -- see README for usage, e.g. `docker run -it noda bash`.
CMD ["bash"]
