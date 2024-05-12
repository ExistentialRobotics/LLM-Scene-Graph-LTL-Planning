#!/usr/bin/env bash

set -e
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname $SCRIPT_DIR)"
OS_NAME=$(source /etc/os-release && echo $NAME)

if [[ "$OS_NAME" == "Ubuntu" ]]; then
    bash $SCRIPT_DIR/install_ubuntu.bash
elif [[ "$OS_NAME" == "Arch Linux" ]]; then
    bash $SCRIPT_DIR/install_archlinux.bash
else
    echo "Unsupported OS: $OS_NAME"
    exit 1
fi
