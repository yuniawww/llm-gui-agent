# My Grounding Agent for OSWorld

This repository contains the implementation of `my_grounding_agent`, an agent designed to operate and be evaluated within the [OSWorld benchmark](https://github.com/xlang-ai/OSWorld.git) environment. The agent leverages the Gemini API for its reasoning capabilities.

## 📋 Table of Contents
* [Installation & Setup](#-installation--setup)
* [API Configuration](#-api-configuration)
* [Evaluation](#-evaluation)

## ⚙️ Installation & Setup

This section guides you through setting up the necessary environment and dependencies to run the agent.

### Prerequisites
Before you begin, ensure you have the following installed:
- [Git](https://git-scm.com/)
- [Python](https://www.python.org/downloads/) (version 3.9 or higher)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Recommended for environment management)

### Step-by-Step Installation

Follow these steps to prepare your environment.

**1. Clone This Repository**

First, clone this repository to your local machine and navigate into the project directory.
```bash
git clone https://github.com/your-username/your-agent-repo.git
cd your-agent-repo
```

**2. Set up the Python Environment**

We strongly recommend using Conda to create an isolated and clean environment. This prevents conflicts with other projects.

```bash
# Create a new conda environment named 'osworld-agent' with Python 3.9
conda create -n osworld-agent python=3.9 -y

# Activate the newly created environment
conda activate osworld-agent

# Install all the required Python packages
pip install -r requirements.txt
```
> **Note:** If you choose not to use Conda, you can manually install the dependencies via `pip install -r requirements.txt`, but ensure your global Python version is >= 3.9.

**3. Set up the OSWorld GUI Environment**

Our agent is designed to run within the OSWorld benchmark. You need to clone and set up the OSWorld environment separately.

```bash
# Clone the official OSWorld repository (preferably in a sibling directory)
git clone https://github.com/xlang-ai/OSWorld.git
```
After cloning, please **follow the detailed setup instructions** provided in the official [OSWorld repository](https://github.com/xlang-ai/OSWorld.git) to prepare the benchmark environment, including any necessary system dependencies or virtual machine setups.

## 🔑 API Configuration

This agent requires access to the Google Gemini API. You must configure your API key and the desired model name as environment variables.

You can set these variables in two ways:

#### Method 1: Using an `.env` file (Recommended)
1.  Create a file named `.env` in the root directory of **this project**.
2.  Add your credentials to the file as follows:

    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    GEMINI_MODEL_NAME="gemini-1.5-flash-latest"
    ```
3.  The application will automatically load these variables. Make sure your `.gitignore` file contains `.env` to prevent accidentally committing your secret keys.

#### Method 2: Exporting in your Shell
You can export the variables directly in your terminal session. Note that this is temporary and will only last for the current session.

```bash
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
export GEMINI_MODEL_NAME="gemini-1.5-flash-latest"
```

-   `GEMINI_API_KEY`: Your secret API key for accessing Google Gemini services.
-   `GEMINI_MODEL_NAME`: The specific model you want to use (e.g., `gemini-pro`, `gemini-1.5-flash-latest`).

## 🚀 Evaluation

To evaluate the performance of `my_grounding_agent.py` on the OSWorld benchmark, follow these steps.

**1. Place Agent Files into the OSWorld Benchmark**

You need to copy the agent logic and the evaluation script into the appropriate directories within the `OSWorld` project you cloned earlier.

Assuming your project folder (`your-agent-repo`) and the `OSWorld` folder are in the same parent directory:
```bash
# Navigate to your agent's directory first if you are not already there
# cd your-agent-repo

# Copy the agent definition to the OSWorld agents directory
cp my_grounding_agent.py ../OSWorld/benchmark/agents/

# Copy the evaluation runner script to the root of the OSWorld benchmark directory
cp run_my_agent.py ../OSWorld/benchmark/
```
> **Important:** Adjust the paths (`../OSWorld/`) if your directory structure is different.

**2. Run the Evaluation Script**

Now, navigate to the OSWorld benchmark directory and execute the runner script.

```bash
# Change directory to the OSWorld benchmark folder
cd ../OSWorld/benchmark/

# Activate the conda environment if it's not already active
conda activate osworld-agent

# Run the evaluation
python run_my_agent.py
```

The script will start the evaluation process. Results, logs, and performance scores will be saved to the output directories as configured within the `run_my_agent.py` script.