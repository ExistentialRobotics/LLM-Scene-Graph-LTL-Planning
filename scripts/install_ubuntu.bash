#!/usr/bin/env bash

set -e
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname $SCRIPT_DIR)"
OS_NAME=$(source /etc/os-release && echo $NAME)

if [[ $OS_NAME != "Ubuntu" ]]; then
    echo "This script is intended to be run on Ubuntu"
    exit 1
fi

sudo apt update
sudo apt install -y git build-essential cmake curl wget tar \
    python3 python3-dev python3-pip python3-urllib3 \
    libeigen3-dev \
    libomp-dev \
    libopencv-dev \
    libboost-all-dev \
    libgraphviz-dev \
    libyaml-cpp-dev \
    libabsl-dev
pip3 install pipenv  # pipenv provided by APT is outdated
export PATH=$PATH:/usr/local/bin:$HOME/.local/bin  # pipenv installed to ~/.local/bin or /usr/local/bin

# create python virtual environment
cd $PROJECT_DIR
pipenv --rm || true
pipenv install --verbose
pipenv shell
PYTHON_SITE_PACKAGES=$(python -c "import site;print(site.getsitepackages()[0])")

# install spot
cd $PROJECT_DIR
mkdir -p build/spot
cd build/spot
if [[ ! -d spot-2.11.6 ]]; then
    if [[ ! -f spot-2.11.6.tar.gz ]]; then
        wget http://www.lrde.epita.fr/dload/spot/spot-2.11.6.tar.gz
    fi
    tar -xvzf spot-2.11.6.tar.gz
fi
cd spot-2.11.6
./configure --prefix /usr --with-pythondir=$PYTHON_SITE_PACKAGES
make -j`nproc`
sudo make install

# Library versions: 05-09-2024
# python3: 3.8 (20.04), 3.10 (22.04)  <---- Ubuntu 20.04 is not supported, Python >= 3.10 is required
# libeigen3-dev: 3.3.7-2 (20.04), 3.4.0-2 (22.04)
# libomp-dev: 1:10.0-50 (20.04), 1:14.0-55 (22.04)
# libopencv-dev: 4.2.0 (20.04), 4.5.4 (22.04)
# libboost-all-dev: 1.17.0 (20.04), 1.74.0.3 (22.04)
# libyaml-cpp-dev: 0.6.2-4 (20.04), 0.7.0 (22.04)
# libabsl-dev: build-from-source (20.04), 20210324.2-2 (22.04)
# spot: build-from-source

# install Python source
cd $PROJECT_DIR
echo "Installing Python source for $(which python)"
pip install -e .

# build C++ source
cd $PROJECT_DIR
echo "Building C++ source"
mkdir -p build/cpp
cd build/cpp
cmake ../..
cmake --build . --target amra_ltl_scene_graph -- -j`nproc`
