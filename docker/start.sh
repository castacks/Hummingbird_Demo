#!/bin/bash

usage() {
    echo "Usage: $0 -n <container_name> -s <source_dir>"
    echo "  -t <container_name> : Name of the container"
    exit 1
}

while getopts "n:" opt; do
    case $opt in
        n)
            CONTAINER_NAME=$OPTARG
            ;;
        *)
            usage
            ;;
    
    case $opt in
        s)
            SOURCE_DIR=$OPTARG
            ;;
        *)
            usage
            ;;
    esac
done

if [ -z "$CONTAINER_NAME" ]; then
    usage
fi

# Check if the container is already running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Container $CONTAINER_NAME is already running"
    exit 0
fi

# Check if the container exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Container $CONTAINER_NAME exists but is not running"
    echo "Starting container $CONTAINER_NAME"
    docker start $CONTAINER_NAME
    exit 0
fi

# Start the container
echo "Starting container $CONTAINER_NAME"
docker run --runtime nvidia -it --privileged --ipc=host --pid=host -e NVIDIA_DRIVER_CAPABILITIES=all -e DISPLAY \
  -v /dev:/dev \
  -v /tmp:/tmp \
  -v /var/nvidia/nvcam/settings/:/var/nvidia/nvcam/settings/ \
  -v /etc/systemd/system/zed_x_daemon.service:/etc/systemd/system/zed_x_daemon.service \
  -v ${HOME}/zed_docker_ai/:/usr/local/zed/resources/ \
  -v ${SOURCE_DIR}:/root/ros2_ws/ \
  $CONTAINER_NAME