SHELL=/bin/bash

docker-cpu:
	bash ./scripts/build_docker.sh

docker-gpu:
	USE_GPU=True bash ./scripts/build_docker.sh

docker-run-cpu:
	bash ./scripts/run_docker_cpu.sh "ls; pwd; /bin/bash"
	
docker-run-gpu:
	bash ./scripts/run_docker_gpu.sh "ls; pwd; /bin/bash"

test-nvidia-gpu:
	sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi