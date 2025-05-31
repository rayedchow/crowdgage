#!/bin/bash
# Set OpenMP environment variable to ignore duplicate libraries
export KMP_DUPLICATE_LIB_OK=TRUE

# Run the audio script
python audio.py
