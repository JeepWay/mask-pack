#!/bin/bash
# Launch an container using the docker gpu image

# Store all parameters passed to this script into the variable `cmd_line`
cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line

docker run -it --rm --gpus=all --volume $(pwd):/home/user/maskpack \
    jeepway/maskpack-gpu:latest bash -c "cd /home/user/maskpack && $cmd_line"

# docker run -it --rm --gpus=all --network host --ipc=host \
#     --mount src=$(pwd),target=/home/user/maskpack,type=bind \
#     jeepway/maskpack-gpu:latest bash -c "cd /home/user/maskpack && $cmd_line"

# --mount: Any operations performed on the /home/user/maskpack directory inside the container will actually affect the $(pwd) directory on the host machine.
# Usage: bash ./scripts/run_docker_cpu.sh "ls; pwd; /bin/bash"

