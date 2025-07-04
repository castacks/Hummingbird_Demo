# #!/bin/bash

# Install ZED SDK
sudo apt update
sudo apt install -y terminator

sudo apt install -y zstd libqt5network5 libqt5opengl5 libqt5sql5 libqt5xml5 cuda nvidia-tensorrt nvidia-tensorrt-dev

sudo apt --reinstall install nvidia-jetpack
wget -O ZED_SDK.run https://download.stereolabs.com/zedsdk/5.0/l4t36.4/jetsons
chmod +x ZED_SDK.run
./ZED_SDK.run -- silent
rm ZED_SDK.run

echo "xhost +" >> ~/.bashrc

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo groupadd docker > /dev/null 2>&1 || true
sudo usermod -aG docker $USER
newgrp docker > /dev/null 2>&1 || true

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

sudo pip3 install -U jetson-stats

# Network tuning

touch ./tmp.conf
echo "net.ipv4.ipfrag_time=3" >> ./tmp.conf
echo "net.ipv4.ipfrag_high_thresh=134217728" >> ./tmp.conf
echo "net.core.rmem_max=2147483647" >> ./tmp.conf
sudo mv ./tmp.conf /etc/sysctl.d/10-rti.conf
sudo sysctl -p /etc/sysctl.d/10-rti.conf

# generate ssh key for github
EMAIL="tharp@andrew.cmu.edu"
KEY_PATH="$HOME/.ssh/id_rsa"

# Check if the key already exists
if [[ -f "$KEY_PATH" ]]; then
    echo "SSH key already exists at $KEY_PATH. Skipping key generation."
else
    # Generate SSH key (no passphrase, overwrite without asking)
    ssh-keygen -t rsa -b 4096 -C "$EMAIL" -f "$KEY_PATH" -N ""
    echo "SSH key generated successfully."
fi

# Start SSH agent if not already running
eval "$(ssh-agent -s)"

# Add private key to SSH agent
ssh-add "$KEY_PATH"

# Display public key
echo "Here is your SSH public key:"
cat "$KEY_PATH.pub"


