# #!/bin/bash

# Install ZED SDK
sudo apt update
sudo apt install -y terminator
sudo apt install -y zstd libqt5network5 libqt5opengl5 libqt5sql5 libqt5xml5 cuda
wget -O ZED_SDK.run https://download.stereolabs.com/zedsdk/4.2/l4t36.4/jetsons
chmod +x ZED_SDK.run
./ZED_SDK.run --silent

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

sudo apt-get install -y docker-ce=5:27.5.1-1~ubuntu.22.04~jammy --allow-downgrades
sudo apt-get install -y docker-ce-cli=5:27.5.1-1~ubuntu.22.04~jammy --allow-downgrades

sudo groupadd docker
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


