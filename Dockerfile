FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY memengine ./memengine
COPY prompts ./prompts
RUN pip install --no-cache-dir '.[all]'
EXPOSE 8000
CMD ["memengine", "serve", "--host", "0.0.0.0"]
