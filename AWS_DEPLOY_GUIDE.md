# Natively Deploying the AI Dashboard onto AWS EC2

This guide will walk you through launching an AWS server, getting your code onto it, and running the included `deploy.sh` script to get everything online.

---

### Step 1: Launch an EC2 Instance on AWS
1. Go to the [AWS Management Console](https://console.aws.amazon.com/) and search for **EC2**.
2. Click **Launch Instance**.
3. **Name:** Give your server a name (e.g., `reddit-ai-dashboard`).
4. **OS / AMI:** Select **Ubuntu** (Ubuntu Server 22.04 LTS or 24.04 LTS is perfect).
5. **Instance Type:** 
   > [!IMPORTANT]
   > Do **NOT** select `t2.micro`. Choose an instance with at least **16GB RAM** (like `t3.xlarge` or `m5.xlarge`). Alternatively, a GPU instance (`g4dn.xlarge`) will run the AI summaries *much* faster.
6. **Key Pair:** Create a new key pair (e.g., `my-aws-key`). This downloads a `.pem` file. **Keep this safe**, you need it to connect!

### Step 2: Configure Network & Security Group
While still on the launch screen, look for **Network Settings** and make sure you do the following:
1. Check **Allow SSH traffic** from **Anywhere**.
2. Click **Edit** network settings, and add a **Custom TCP Rule**.
3. Enter Port **8501** and set Source to **0.0.0.0/0**. *(This allows the world to see your Streamlit dashboard).*

Now click **Launch Instance**.

---

### Step 3: Connect to your Server
1. Go to your EC2 dashboard, select your running instance, and copy its **Public IPv4 address**.
2. Open your local terminal/command prompt and connect via SSH:
   ```bash
   # If you are on Windows, standard Command Prompt or PowerShell works!
   # Make sure you provide the path to where you saved your .pem file
   ssh -i "path/to/your/my-aws-key.pem" ubuntu@<YOUR_PUBLIC_IP>
   ```

---

### Step 4: Run the Deployment Script!
Once you are logged into the AWS server terminal (you should see `ubuntu@ip-172-...`), clone your code and run the deployment script:

```bash
# 1. Clone your project code
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Make the script executable and run it
chmod +x deploy.sh
./deploy.sh
```
*The script takes about 5-10 minutes. It installs Python, downloads Ollama, downloads the `llama3.2` model, and installs all the python packages from your `requirements.txt`!*

---

### Step 5: Start the Dashboard
Once the script is finished, it will provide instructions on how to use `tmux` (a tool that keeps apps running after you close the SSH window). Just type:

```bash
tmux new -s dashboard
source venv/bin/activate
streamlit run app.py --server.port 8501
```

> **To safely exit the terminal without closing the app:** Press `Ctrl` + `B`, then let go and press `D`.

### That's it! 🎉
Open your browser and navigate to `http://<YOUR_PUBLIC_IP>:8501`. Your dashboard is live, utilizing a natively hosted Ollama model!
