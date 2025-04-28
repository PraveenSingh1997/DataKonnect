# ─── Builder Stage ─────────────────────────────────────────────────────────────
# Use a slightly older Python 3.12 image with a more standard Debian base if
# needed, though slim is usually fine. Sticking with slim for now.
FROM python:3.12-slim AS builder

# Set a user to avoid running as root
USER root

WORKDIR /app

# Install build dependencies
# Combine RUN commands to reduce layers
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && pip install --no-cache-dir --upgrade pip

# Copy requirements and wheel dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# ─── Final Stage ──────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Set a user to avoid running as root
USER root

WORKDIR /app

# install runtime deps: netcat (for health checks/waiting) + Postgres client
# Combine RUN commands and clean up apt cache
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
           netcat-openbsd \
           postgresql-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy built wheels and install dependencies without internet
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy the rest of the application code
COPY . .

# Command to run the application
# Consider using an entrypoint script for pre-launch tasks like waiting for DB
CMD ["python", "run.py"]