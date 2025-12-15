#!/bin/bash

python -m venv $HOME/qdk_chemistry_venv
source $HOME/qdk_chemistry_venv/bin/activate

pip install --upgrade pip

# Install node
export NVM_DIR="$HOME/.nvm"
mkdir -p $NVM_DIR
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source $NVM_DIR/nvm.sh && nvm install node && nvm use node

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
which rustc

# Install ipykernel
pip install ipykernel ipywidgets pandas

# Install QDK Chemistry package
cd "$REPO_ROOT/python"
export CMAKE_BUILD_PARALLEL_LEVEL=4
QDK_UARCH=native pip install -v .[all]

# Install QDK Widgets
git clone --branch billti/widg https://github.com/microsoft/qdk /tmp/qdk
cd /tmp/qdk
python ./prereqs.py --install
./build.py --widgets --qdk --pip
pip install --force-reinstall ./target/wheels/*.whl
rm -rf /tmp/qdk

# Add venv to bash
echo "source $HOME/qdk_chemistry_venv/bin/activate" >> $HOME/.bashrc
