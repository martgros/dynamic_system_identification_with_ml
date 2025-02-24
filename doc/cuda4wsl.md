## Use CUDA for WSL

I am using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) on my windows pc to run the code in this repository. In the following a short hint is given how to install CUDA (to use GPU if you have one on your PC).

### install NVIDIA App

install from [here](https://www.nvidia.com/en-us/drivers/) and install necessary GPU driver (if not anyhow installed).


[install instructions:](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

sudo apt update
sudo apt install libcudnn8 libcudnn8-dev



[install tensorflow for wsl](https://www.tensorflow.org/install/pip#windows-wsl2)

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.7.1/local_installers/cudnn-local-repo-ubuntu2204-9.7.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.7.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.7.1/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
```

futher needed? (nvcc --version)
`sudo apt install nvidia-cuda-toolkit`