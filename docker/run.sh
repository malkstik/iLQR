mountdir="$1"
IMAGE_TAG="$2"

CONTAINER_NAME=optimal-control

DOCKER_OPTIONS=""
DOCKER_OPTIONS+="-it "
DOCKER_OPTIONS+="--name $CONTAINER_NAME-$IMAGE_TAG "
DOCKER_OPTIONS+="-v $mountdir:/home/malkstik/mnt "
DOCKER_OPTIONS+="--net=host "
DOCKER_OPTIONS+="--privileged "

# X11 forwarding for GUI
xhost local:root
export DISPLAY=$DISPLAY
DOCKER_OPTIONS+="-e DISPLAY=$DISPLAY "
DOCKER_OPTIONS+="-v /tmp/.X11-unix:/tmp/.X11-unix "
# DOCKER_OPTIONS+="-v $HOME/.Xauthority:/home/$(whoami)/.Xauthority "

echo "docker run" $DOCKER_OPTIONS $CONTAINER_NAME":"$IMAGE_TAG /bin/bash 
docker run $DOCKER_OPTIONS $CONTAINER_NAME":"$IMAGE_TAG  /bin/bash 
