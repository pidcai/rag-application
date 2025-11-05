#!/bin/bash
set -e

echo "Starting Streamlit deployment..."

echo "Stopping existing Streamlit app..."
sudo fuser -k 8501/tcp || true

echo "Installing dependencies..."
sudo apt update -y
sudo apt install -y python3 python3-venv python3-pip git curl

echo "Updating repository..."
if [ -d "rag-application" ]; then
  cd rag-application
  git reset --hard
  git pull
else
  git clone https://github.com/pidcai/rag-application.git
  cd rag-application
fi

echo "Setting up Python virtual environment..."
if [ ! -d "env" ]; then
  python3 -m venv env
fi
source env/bin/activate

pip install vllm

vllm serve meta-llama/Llama-3.2-3B-Instruct --download-dir /home/models --max-model-len 4096

echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Launching Streamlit app..."
nohup streamlit run streamlit_ui.py > streamlit.log 2>&1 &

echo "Streamlit deployment triggered successfully!"