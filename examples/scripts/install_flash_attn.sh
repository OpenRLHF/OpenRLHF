# install flash attention2.0
git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git
export NVTE_FRAMEWORK=pytorch   # Optionally set framework
pip install --user ./TransformerEngine # Build and install

pip install --user https://github.com/Dao-AILab/flash-attention/releases/download/v2.1.1/flash_attn-2.1.1+cu121torch2.1cxx11abiTRUE-cp310-cp310-linux_x86_64.whl