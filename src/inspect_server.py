"""Check that the LM Studio server is reachable and list the loaded models.

Usage:
    python src/inspect_server.py
    python src/inspect_server.py --base-url http://localhost:1234/v1

Run from the repository root.
"""

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import argparse

from shared.lmstudio_client import DEFAULT_BASE_URL, build_client
from shared.reporting import OK, ERR, print_header, print_status

# ─────────────────────────────────────────────────────────────
# Constants and defaults
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "base_url": DEFAULT_BASE_URL,
}

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inspect a local LM Studio server.")
    parser.add_argument("--base-url", default=DEFAULTS["base_url"])
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def list_model_ids(client):
    """Return the model ids the server currently exposes."""
    listing = client.models.list()
    return [entry.id for entry in listing.data]

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print_header(f"Server inspection: {args.base_url}")
    client = build_client(args.base_url)

    try:
        model_ids = list_model_ids(client)
    except Exception as exception:
        print_status(ERR, f"could not reach server: {type(exception).__name__}: {exception}")
        print_status(ERR, "is the server running in the Developer tab on this port?")
        return

    print_status(OK, f"server reachable, {len(model_ids)} model(s) loaded")
    for model_id in model_ids:
        print_status(OK, f"model id: {model_id}")


if __name__ == "__main__":
    main()
