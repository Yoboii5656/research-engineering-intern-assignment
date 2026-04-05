#!/bin/bash
# ==========================================
# AWS EC2 Ubuntu Deployment Script
# Automatically sets up Streamlit & Ollama
# ==========================================

set -e # Exit immediately if a command exits with a non-zero status

echo "🚀 Starting Deployment Setup..."

# 1. Update system and install Python/pip/tmux
echo "📦 Updating system dependencies & installing Python 3 + tmux..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv tmux curl git

# 2. Install Ollama Native
echo "🦙 Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama service to ensure it's running
echo "⚙️  Starting Ollama Service..."
sudo systemctl enable ollama
sudo systemctl start ollama

# 3. Pull required AI Model (llama3.2)
echo "🧠 Downloading llama3.2 model (this may take a few minutes)..."
# We sleep briefly to ensure the service is fully up before pulling
sleep 5 
ollama pull llama3.2

# 4. Setup Python Virtual Environment and install requirements
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "📥 Installing python packages..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found! Skipping..."
fi

echo ""
echo "✅ Setup Complete!"
echo ""
echo "🔥 To start the server and keep it running 24/7, run the following commands:"
echo "--------------------------------------------------------"
echo "  1. Start a persistent session:  tmux new -s dashboard "
echo "  2. Activate environment:        source venv/bin/activate "
echo "  3. Run Streamlit:               streamlit run app.py --server.port 8501"
echo "  4. Detach session:              Press Ctrl+B, then D"
echo "--------------------------------------------------------"
echo "Your app will be available at: http://<YOUR_EC2_PUBLIC_IP>:8501"
