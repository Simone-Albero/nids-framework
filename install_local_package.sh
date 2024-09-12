#!/bin/bash

PACKAGE_NAME="nids_framework"
PACKAGE_DIR="nids_framework"

cd "$PACKAGE_DIR" || exit
echo "Installing $PACKAGE_NAME ..."
pip install .

if [ $? -eq 0 ]; then
  echo "Package $PACKAGE_NAME downloaded successfully."
else
  echo "An error occurred while installing the package $PACKAGE_NAME."
  exit 1
fi
