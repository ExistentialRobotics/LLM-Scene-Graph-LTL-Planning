#!/usr/bin/env bash

set -e
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname $SCRIPT_DIR)"
OS_NAME=$(source /etc/os-release && echo $NAME)

if [[ $OS_NAME != "Arch Linux" ]]; then
    echo "This script is intended to be run on Arch Linux"
    exit 1
fi

sudo pacman -Syyu --noconfirm
sudo pacman -S --noconfirm git base-devel cmake curl wget tar \
    python python-pip python-pipenv \
    eigen \
    openmp \
    opencv qt6-base \
    boost \
    graphviz \
    yaml-cpp \
    abseil-cpp

# 05-09-2024: ray does not support Python 3.12 yet
# let's get Python 3.10
# we use paru to install Python 3.10 from AUR, you can use your favorite AUR helper
cd $PROJECT_DIR
mkdir -p build
cd build
git clone https://aur.archlinux.org/paru.git
cd paru
makepkg -sci
paru -S python310 python310-pip

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
