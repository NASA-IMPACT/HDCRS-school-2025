#!/bin/bash
# ESA-NASA Workshop Environment Setup Script
# This script sets up the required conda environments for the ESA-NASA workshop

set +e

# Configuration variables
REPOSITORY_NAME="HDCRS-school-2025"
REPOSITORY_URL="https://github.com/NASA-IMPACT/${REPOSITORY_NAME}.git"
REPOSITORY_PATH="/home/sagemaker-user/${REPOSITORY_NAME}"

# Log messages with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Clone repository if it doesn't exist already
clone_repository() {
    if [[ ! -d "$REPOSITORY_PATH" ]]; then
        log "Cloning repository from $REPOSITORY_URL"
        git -C /home/sagemaker-user clone "$REPOSITORY_URL"
        if [[ $? -ne 0 ]]; then
            log "ERROR: Failed to clone repository"
            return 1
        fi
        log "Repository cloned successfully"
    else
        log "Repository already exists at $REPOSITORY_PATH"
    fi
    return 0
}


# Create and setup conda environment
create_conda_env() {
    local env_name="$1"

    log "Setting up conda environment: $env_name"

    # Check if environment already exists
    if conda info --envs | grep -q "$env_name"; then
        log "Environment '$env_name' already exists"
    else
        log "Creating conda environment '$env_name'"
        conda create -n "$env_name" python=3.12 -y -q

        if [[ $? -ne 0 ]]; then
            log "ERROR: Failed to create conda environment '$env_name'"
            return 1
        fi

        # Activate the new environment and install packages
        log "Installing packages for '$env_name'"
        conda activate "$env_name"

        # Install ipykernel in all environments
        pip install ipykernel

        # Install requirements from the appropriate file
        local requirements_file="${REPOSITORY_PATH}/environments/${env_name}/requirements.txt"
        if [[ -f "$requirements_file" ]]; then
            pip install -r "$requirements_file"
            if [[ $? -ne 0 ]]; then
                log "WARNING: Some packages in requirements.txt may have failed to install"
            fi
        else
            log "WARNING: Requirements file not found at $requirements_file"
        fi

        # Special setup for weather environment
        if [[ "$env_name" == "prithvi_wx" ]]; then
            setup_weather_environment
        fi

        # Register the kernel
        python -m ipykernel install --user --name "$env_name" --display-name "$env_name"

        conda deactivate
    fi

    log "Setup complete for environment: $env_name"
    return 0
}

# Main function to orchestrate the setup
main() {
    log "Starting workshop environment setup"

    # Ensure conda is available
    if ! command_exists conda; then
        log "ERROR: conda is not installed or not in PATH"
        return 1
    fi

    # Activate base conda environment
    log "Activating base conda environment"
    source /opt/conda/bin/activate

    # Reinstall libsqlite (as in the original script)
    log "Reinstalling libsqlite"
    conda install libsqlite --force-reinstall -y

    # Clone the repository
    clone_repository || return 1

    # Check if environments directory exists
    local environments_dir="${REPOSITORY_PATH}/environments"
    if [[ ! -d "$environments_dir" ]]; then
        log "ERROR: Environments directory not found at $environments_dir"
        return 1
    fi

    # Setup each environment
    log "Setting up all environments found in $environments_dir"
    for env_dir in "$environments_dir"/*; do
        if [[ -d "$env_dir" ]]; then
            env_name=$(basename "$env_dir")
            create_conda_env "$env_name"
        fi
    done

    jupyter labextension install jupyter-leaflet

    jupyter lab build

    log "Workshop environment setup completed successfully"
    return 0
}

# Run the main function
main

# Exit on error, but allow for unbound variables
set -e
exit $?
