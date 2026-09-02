# Small base: fewer megabytes, fewer CVEs than the full python image.
# 3.12 matches the development environment on purpose — dev/prod version drift
# is its own source of bugs, and tomllib (used by config.py) needs 3.11+.
FROM python:3.12-slim

# Don't write .pyc files; don't buffer stdout (so logs appear in `docker logs`).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Created before COPY so --chown below can name it.
RUN useradd --create-home appuser

# Copy requirements FIRST, install, THEN copy source. Docker caches each layer,
# so putting source first would reinstall every dependency on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --chown because COPY preserves host permission bits as-is and root-owns the
# result. Files that are not world-readable on the build machine would then be
# unreadable to the non-root USER at runtime — which is exactly the
# PermissionError this image hit before the flag was added. With --chown the
# image is correct regardless of the umask on whoever built it.
COPY --chown=appuser:appuser ticket_agent/ ./ticket_agent/

# config.toml is CONFIGURATION, so it ships in the image: it decides what the
# agent does and is version-controlled for that reason. Secrets do the opposite
# — .env is gitignored, never copied here, and the keys arrive at run time:
#   docker run --rm -e GROQ_API_KEY=$GROQ_API_KEY ticket-agent --file /data/t.txt
COPY --chown=appuser:appuser config.toml .

# Don't run as root. If the container is compromised, this limits the damage.
USER appuser

ENTRYPOINT ["python", "-m", "ticket_agent"]

# No arguments by default: --help is a safer thing for a bare `docker run` to do
# than silently waiting on stdin for a ticket that will never arrive.
CMD ["--help"]
