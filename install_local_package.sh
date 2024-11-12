#!/bin/bash

PACKAGE_NAME="nids_framework"
PACKAGE_DIR="nids_framework"
VENV_DIR=${1:-venv}

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "Upgrading pip to the latest version..."
pip3 install --upgrade pip

cd "$PACKAGE_DIR" || exit

echo "Installing $PACKAGE_NAME in virtual environment $VENV_DIR ..."
pip install .

if [ $? -eq 0 ]; then
  echo "Package $PACKAGE_NAME installed successfully in virtual environment $VENV_DIR."
else
  echo "An error occurred while installing the package $PACKAGE_NAME."
  deactivate
  exit 1
fi

deactivate
