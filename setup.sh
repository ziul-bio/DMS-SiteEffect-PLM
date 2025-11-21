#!/bin/bash

#chmod -R a+rx checkpoints


# Project and Venv paths (dynamically determined based on current directory)
PROJECT_PATH="$(pwd)"
VENV_PATH="$PROJECT_PATH/venv"
ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"

# Ensure virtual environment exists and create it if necessary
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv "$VENV_PATH"
else
    echo "Virtual environment already exists."
fi

# Install dependencies if requirements.txt exists
if [ -f "$PROJECT_PATH/requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    source "$ACTIVATE_SCRIPT"
    pip install --upgrade pip
    pip install -r "$PROJECT_PATH/requirements.txt"
else
    echo "requirements.txt not found, skipping package installation."
fi

# Set up PYTHONPATH in activate script if not already set
PYTHONPATH_LINE='export PYTHONPATH="'"$PROJECT_PATH"':$PYTHONPATH"'

if grep -Fxq "$PYTHONPATH_LINE" "$ACTIVATE_SCRIPT"; then
    echo "PYTHONPATH already set in activate script."
else
    echo "" >> "$ACTIVATE_SCRIPT"
    echo "# PYTHONPATH added during setup!" >> "$ACTIVATE_SCRIPT"
    echo "$PYTHONPATH_LINE" >> "$ACTIVATE_SCRIPT"
    echo "PYTHONPATH added to activate script."
fi

echo "Setup complete!"
