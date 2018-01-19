#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

echo "Download resnet18 trained on Places365"
echo "Downloading $MODEL"
mkdir -p zoo
pushd zoo
wget --progress=bar \
   http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar
popd

echo "done"
