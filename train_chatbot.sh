#!/bin/bash

# Usage
# train_chatbot.sh path_to_virtual_env/activate.sh path_to_chatbot/chatbot.py path_to_data/

# Print hostname and environment
#hostname
#env
  
# Print directory name  
pwd 
echo "set encoding"
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

# Print input parameters
echo "virtual env activation script"  
echo "$1"
echo "chatbot.py"
echo "$2"
echo "data"
echo "$3"

# Set up virtual environment with dependencies 
source $1

# Train model
echo "begin training..."

python3 "$2" "$3"

echo "training finished"
