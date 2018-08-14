#!/bin/bash
# Stop on error
set -e

CONDA_ENV=tf_binding_challenge

if which conda; then
  echo "=== Found Conda ($(conda --version))."
else
  echo "=== Conda does not exist on your system. Please install Conda first."
  echo "https://conda.io/docs/user-guide/install/index.html#regular-installation"
  exit 1
fi

if conda env list | grep -wq ${CONDA_ENV}; then
  echo "=== Removing Conda env (${CONDA_ENV})..."
  conda env remove -n ${CONDA_ENV} -y
else
  echo "=== Conda env (${CONDA_ENV}) does not exist or has already been removed."
fi
