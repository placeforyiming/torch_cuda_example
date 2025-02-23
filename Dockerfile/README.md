
Note:

build the docker image from the dockerfile:

docker build -t cuda:latest .


create the docker container by the following command:

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -p 8888:8888 -p 6006:6006 --name dl --mount type=bind,source=/home/user/workspace,target=/workspace --env="DISPLAY" --env="QT_X11_NO_MTTSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" cuda:latest bash


run the following command after getting into the docker container:

apt purge -y libhwloc-dev libhwloc-plugins
