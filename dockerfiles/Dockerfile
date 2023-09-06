FROM python:3.10

ENV PLATFORM=cu118
ENV TORCH_VERSION=2.0.1

RUN apt-get update
RUN apt-get install libpq-dev gcc less curl unzip libxrender1 libsm-dev libxext-dev -y
# Install Poetry
RUN curl -sSL https://install.python-poetry.org | \
    POETRY_HOME=/opt/poetry python3 && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false


WORKDIR /app/
# Copy dependency related files
COPY ./pyproject.toml ./poetry.lock /app/
# RUN --mount=type=cache,target=/root/.cache/pypoetry/cache \
#     --mount=type=cache,target=/root/.cache/pypoetry/artifacts \
#     poetry install --no-root

COPY ./deps.py /app/deps.py

RUN poetry install --no-root --no-cache --without dev
RUN python -m deps
RUN poetry cache clear --no-interaction --all pypi

COPY ./mariner /app/mariner
COPY ./cli /app/cli
COPY ./api /app/api
COPY ./fleet /app/fleet
COPY ./oauth_providers /app/oauth_providers

RUN poetry install

RUN python -m deps

COPY ./alembic.ini ./pylintrc ./pytest.ini /app/
COPY ./scripts /app/scripts

CMD [ "sh", "/app/scripts/prestart.sh", "&&", "poetry run python -m api.main"]

HEALTHCHECK --retries=10\
    CMD curl -f http://localhost:8000/health || exit 1

