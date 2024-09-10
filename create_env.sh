#!/bin/bash

# Function to install packages using pip
install_packages() {
    pip install torch
    pip install git+https://github.com/facebookresearch/esm.git
    pip install git+https://github.com/songlab-cal/gpn.git
    pip install matplotlib seaborn scikit-learn tensorboard ipykernel pandas
}

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found, proceeding with virtualenv setup"
    env_choice="virtualenv"
else
    # To conda or (not to conda) virtualenv
    echo "Which environment manager would you like to use?"
    echo "1) Conda"
    echo "2) Virtualenv"
    read -p "Enter your choice (1 or 2): " env_choice
fi

# Get environment name
read -p "Enter the environment name: " env_name

# Create and activate Conda environment
if [ "$env_choice" == "1" ] || [ "$env_choice" == "conda" ]; then
    conda create --name "$env_name" python=3.10.12 -y
    source activate "$env_name"

# Create and activate Virtualenv environment
elif [ "$env_choice" == "2" ] || [ "$env_choice" == "virtualenv" ]; then
    python3 -m pip install virtualenv
    python3 -m virtualenv "$env_name"
    source "$env_name/bin/activate"
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Install necessary packages
install_packages

echo "Environment setup and packages installation complete."
