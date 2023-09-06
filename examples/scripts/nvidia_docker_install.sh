set -x

# remove old docker
sudo apt-get autoremove docker docker-ce docker-engine docker.io containerd runc
dpkg -l |grep ^rc|awk '{print $2}' |sudo xargs dpkg -P
sudo apt-get autoremove docker-ce-*
sudo rm -rf /etc/systemd/system/docker.service.d
sudo rm -rf /var/lib/docker

# install docker
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker

# install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker ps