#!/bin/bash
# Stop on error
set -e

CONDA_ENV=tf_binding_challenge
SH_SCRIPT_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)
REQ_TXT=${SH_SCRIPT_DIR}/requirements.txt
if which conda; then
  echo "=== Found Conda ($(conda --version))."
else
  echo "=== Conda does not exist on your system. Please install Conda first."
  echo "https://conda.io/docs/user-guide/install/index.html#regular-installation"
  exit 1
fi

conda create -n ${CONDA_ENV} --file ${SH_SCRIPT_DIR}/requirements.txt -c bioconda -y

source activate ${CONDA_ENV}
  pip install synapseclient
  git clone https://github.com/nboley/grit
  cd grit
  python setup.py install
  cd .. && rm -rf grit
source deactivate
