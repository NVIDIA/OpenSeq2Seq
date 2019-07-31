apt-get update
apt-get install -y python3-pip libhdf5-serial-dev hdf5-tools
pip3 install -U pip
# apt-get remove -y python3-pip
pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu
