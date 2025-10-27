FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    bash

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock ./

COPY . .

RUN poetry install --no-interaction --no-ansi --no-root


CMD ["/bin/bash"]