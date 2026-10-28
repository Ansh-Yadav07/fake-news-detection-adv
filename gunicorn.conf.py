"""Gunicorn configuration. Gunicorn loads this file automatically from the working directory.

Override at runtime with environment variables:
  PORT             - port to bind (default 5001)
  WEB_CONCURRENCY  - number of worker processes (default 1; models are memory heavy)
"""
import os

# Bind to all interfaces on $PORT (default 5001).
bind = f"0.0.0.0:{os.environ.get('PORT', '5001')}"

# Keep to one worker by default: each worker loads the ML models into memory.
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))

# Threads per worker for I/O-bound request handling.
threads = 4

# Seconds before a silent worker is killed and restarted.
timeout = 30

# Load the app once in the master so workers share model memory via copy-on-write.
preload_app = True

# Write access logs to stdout.
accesslog = "-"
