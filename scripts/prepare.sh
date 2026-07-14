#!/bin/bash

sudo apt update 

sudo apt install datacenter-gpu-manager-4-cuda13 datacenter-gpu-manager-4-cuda12 ripgrep poppler-utils -y

curl -fsSL https://github.com/aannoo/hcom/releases/latest/download/hcom-installer.sh | sh