#!/bin/bash

set -e

PACKAGE_NAME="nids_framework"
PACKAGE_DIR="nids_framework"
VENV_DIR=${1:-venv}

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in '$VENV_DIR'..."
  if ! python3 -m venv "$VENV_DIR"; then
    echo "Error: Failed to create the virtual environment in '$VENV_DIR'." >&2
    exit 1
  fi
fi

source "$VENV_DIR/bin/activate"

echo "Upgrading pip to the latest version..."
pip install --upgrade pip

if cd "$PACKAGE_DIR"; then
  echo "Installing '$PACKAGE_NAME' in virtual environment '$VENV_DIR'..."
  if pip install .; then
    echo "Package '$PACKAGE_NAME' installed successfully in virtual environment '$VENV_DIR'."
  else
    echo "Error: Failed to install package '$PACKAGE_NAME'." >&2
    deactivate
    exit 1
  fi
else
  echo "Error: Failed to change directory to '$PACKAGE_DIR'." >&2
  deactivate
  exit 1
fi

deactivate