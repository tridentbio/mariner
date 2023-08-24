FROM python:3.9-slim

WORKDIR /app/

COPY ./mariner/__init__.py /app/mariner/__init__.py
COPY ./api/__init__.py /app/api/__init__.py
# COPY ./model_builder/__init__.py /app/model_builder/__init__.py

COPY ./pyproject.toml ./poetry.lock ./deps.py /app/

# RUN /usr/local/bin/python3 -m pip install --upgrade pip

ENV POETRY_VERSION=1.2.0a2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache
RUN python3 -m venv $POETRY_VENV \
    && pip install -U pip setuptools \
    && pip install poetry
ENV PATH="${PATH}:${POETRY_VENV}/bin"
RUN poetry config virtualenvs.create false

RUN apt-get update
RUN apt-get install libpq-dev gcc less curl unzip libxrender1 -y

RUN poetry install --no-root --no-cache --without dev && poetry cache clear --no-interaction --all pypi

COPY ./mariner /app/mariner
COPY ./cli /app/cli
COPY ./api /app/api
COPY ./fleet /app/fleet
COPY ./scripts /app/scripts
COPY ./tests /app/tests
COPY ./oauth_providers /app/oauth_providers
# COPY ./conf.toml /app/conf.toml
COPY ./alembic.ini /app/alembic.ini
ENV PYTHONPATH=/app

RUN bash -c "poetry run install_deps_${GEO_DEPS-cpu}"

# COPY --from=quay.io/oauth2-proxy/oauth2-proxy:v7.3.0-amd64 /bin/oauth2-proxy /usr/local/bin/oauth2-proxy

COPY ./entrypoints/ /app/entrypoints
RUN chmod +x /app/entrypoints/*
CMD ["sh"]
