#!/bin/bash

# Define the URLs for the zip files
URL1="http://example.com/file1.zip"
URL2="http://example.com/file2.zip"

# Define the directory to save the files
DATA_DIR="data"

# Create the directory if it doesn't exist
mkdir -p $DATA_DIR

# Download the zip files and save them in the data directory
wget -O $DATA_DIR/file1.zip $URL1
wget -O $DATA_DIR/file2.zip $URL2

# Unzip the files in the data directory
unzip $DATA_DIR/file1.zip -d $DATA_DIR
unzip $DATA_DIR/file2.zip -d $DATA_DIR

# Remove the zip files after extraction
rm $DATA_DIR/file1.zip
rm $DATA_DIR/file2.zip

echo "Download, extraction, and cleanup completed. Files are in the '$DATA_DIR' directory."

