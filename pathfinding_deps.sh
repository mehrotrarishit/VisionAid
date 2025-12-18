apt-get update

apt-get install -y libgl1 libglib2.0-0

pip install -e .

pip install opencv-python-headless

pip install opencv-python numpy ultralytics scipy

pip install numpy==1.26.4

echo "[INFO] Starting dependency installation..."

pip install yacs

git clone https://github.com/apple/ml-depth-pro.git .

pip install -e .

source get_pretrained_models.sh

# Clone the repo directly into the current directory
git clone https://github.com/CSAILVision/semantic-segmentation-pytorch.git .

# Initialize git if needed (not required if clone worked, but keeping for safety)
git init
git remote add origin https://github.com/CSAILVision/semantic-segmentation-pytorch.git
git pull origin master

# Download pretrained models (without running demo)
DOWNLOAD_ONLY=1 ./demo_test.sh

echo "[INFO] Dependencies installed successfully."

pip install transformers

#Install accelerate
pip install 'accelerate>=0.26.0'
pip install -U bitsandbytes

pip install perplexityai

pip install 'spacy<3.7'

python -m spacy download en_core_web_sm
