# VisionAid: A Speech-Driven, Scene-Aware AI System for Assisting Visually Impaired Individuals

##  Introduction
**VisionAid** is a fully on-device, speech-driven assistive navigation system designed to help visually impaired individuals by converting monocular visual input into **goal-oriented guidance**. Unlike conventional systems that merely describe the environment, VisionAid interprets the scene, determines traversable space, and delivers step-by-step movement instructions — all through a simple **voice interaction loop**. It unifies speech recognition, intent classification, scene understanding, general and scene-aware query handling, and navigation in a single embedded pipeline that runs entirely on a Jetson device, requiring only an RGB camera. The system bridges perception and motion, offering users not just awareness of surroundings but actionable, real-time guidance for safe mobility.

---

##  Pipeline Overview
- **Input:** User prompt (spoken query such as *“Guide me to the person near me”*)  
- **Output:** Final spoken navigation guidance (directional instructions, obstacle awareness, and contextual path description)  

The pipeline is speech-driven, it listens for a wake-word (“Okay Vision”), transcribes the query, classifies intent, and triggers the appropriate processing pipeline.  
For path-finding queries, it performs:
- Target detection (YOLO12)
- Depth estimation (DepthPro)
- Walkable mask generation (ADE20K-trained segmentation)
- Path planning (A* with inflation-aware traversal)
- RDP-based path simplification and instruction generation  

---

##  Requirements
- **Hardware:** 
  - NVIDIA Jetson AGX Orin Developer Kit
  - Waveshare Audio Card and Speaker
  - GoPro Hero 8 Black
- **Models & Frameworks:**
  - Vosk (Wake-word listener)
  - Whisper (Speech-to-Text)
  - BERT (Intent Classifier)
  - YOLO12 (Target Detection)
  - DepthPro (Monocular Depth Estimation)
  - ADE20K Semantic Segmentation
  - A* Pathfinding + RDP Simplification
  - MMS-TTS (Speech Synthesis)
  - Perplexity Sonar (LLM for contextual response)
- **Languages:** Python 3.8+  
- **Dependencies:** Torch 2.1.0, Torchvision 0.16.1, Transformers 4.40.2, Librosa 0.10.2, SoundFile 0.12.1  

---

##  Jetson Implementation

###  1. Setup (Headless Mode)
Follow NVIDIA’s official headless mode guide:  
[Get Started - Jetson AGX Orin](https://developer.nvidia.com/embedded/learn/get-started-jetson-agx-orin-devkit)

Check JetPack version:
```bash
head -n 1 /etc/nv_tegra_release
```

---

###  2. Wi-Fi Connection
Connect to Wi-Fi:

**Note**: You may have to change the wifi connection command according to your wifi security protocol

```bash
sudo nmcli dev wifi list
sudo nmcli connection add type wifi ifname wlan0 con-name <your_wifi_name> ssid <your_ssid> wifi-sec.key-mgmt wpa-eap 802-1x.eap peap 802-1x.identity <your_username> 802-1x.password <your_username> 802-1x.phase2-auth mschapv2

sudo nmcli connection up <your_wifi_name>
nmcli connection show --active
ip a show wlan0
```

---

###  3. SSH Connection
Connect via terminal:
```bash
ssh username@<Jetson_IP>
```
Ensure you’re connected to the same network.

---

###  4. Create Virtual Environment and Install Requirements
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libopenblas-dev
python3 -m venv e2e_env
source e2e_env/bin/activate
python3 -m pip install --upgrade pip
```

#### PyTorch & Torchvision Installation
```bash
wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
pip install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

git clone https://github.com/pytorch/vision.git
cd vision
git checkout tags/v0.16.1
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
python3 setup.py install
```

#### Additional Packages
```bash
pip install "transformers==4.40.2" "soundfile==0.12.1" "librosa==0.10.2"
```

---

###  5. Verify CUDA and Torch
```bash
nvcc --version
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

###  6. Audio Setup
```bash
sudo apt-get install -y ffmpeg
arecord -l  # Check microphone
aplay -l    # Check speaker
```

---

###  7. GoPro Interfacing 
```bash
git clone https://github.com/jschmid1/gopro_as_webcam_on_linux.git
cd gopro_as_webcam_on_linux
sudo ./install.sh

sudo apt-add-repository universe
sudo apt update
sudo apt install -y nvidia-l4t-kernel-headers
git clone https://github.com/umlaeute/v4l2loopback.git
cd v4l2loopback
make clean && make
sudo make install
sudo depmod -a
sudo modprobe v4l2loopback
sudo gopro webcam -a
ls /dev/video*
```

---

##  Installing Dependencies
The `pathfinding_deps.sh` script installs all dependencies needed for the navigation and segmentation modules.

The pathfinding_deps.sh script installs all essential dependencies for the VisionAid navigation pipeline, including OpenCV, Ultralytics (YOLO), semantic segmentation modules, DepthPro and supporting libraries like Transformers, Perplexity, and Accelerate.

Run the script:
```bash
source pathfinding_deps.sh
```

---

##  Running the Pipeline
1. **Activate virtual environment**
   ```bash
   source e2e_env/bin/activate
   ```
2. **Start GoPro camera (in a separate terminal)**
   ```bash
   sudo gopro webcam -a
   ```
3. **Run the VisionAid pipeline**
   ```bash
   python wake_listener.py
   ```

The system will listen for the wake word (“Okay Vision”) and process user queries for real-time scene-aware navigation.

---

##  Versions Tested
| Component | Version |
|------------|----------|
| JetPack | 5.1.2 |
| Torch | 2.1.0 |
| Torchvision | 0.16.1 |
| Python | 3.8 |
| CUDA | 11.4 |

---


##  Authors
- **Rishit Mehrotra** - Developer
- **Raina Tathed** - Developer
