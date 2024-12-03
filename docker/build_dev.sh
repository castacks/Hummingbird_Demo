#!/bin/bash
cd $(dirname $0)

if [ "$#" -lt 1 ]; then
    echo "Give L4T version then ZED SDK version has parameters, like this:"
    echo "./build_dev.sh l4t-r35.4.1"
    exit 1
fi

# L4T version
# Verify the format (l4t-r digits.digits.digits)
if ! [[ $1 =~ ^l4t-r[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Invalid L4T version format."
    exit 1
fi

L4T_version=$1
# Remove the prefix 'l4t-r'
l4t_version_number="${L4T_version#l4t-r}"

# copy the wrapper content
rm -r ./tmp_sources
mkdir -p ./tmp_sources
cp -r ../zed_src/ ./tmp_sources

# Split the string and assign to variables
IFS='.' read -r l4t_major l4t_minor l4t_patch <<< "$l4t_version_number"
###########

echo "Building dockerfile for $1 and ZED SDK $2"
docker build -t zed_ros2_l4t_image \
--build-arg L4T_VERSION=$1 \
--build-arg L4T_MAJOR=$l4t_major \
--build-arg L4T_MINOR=$l4t_minor \
-f ./Dockerfile.zed-dev .