MAJOR ?= 0
MINOR ?= 1
VERSION = $(MAJOR).$(MINOR)
APP_NAME ?= cs695-005

# Docker args
DISPLAY ?= :0.0
XPASSTHROUGH ?= true
DOCKER_FILE_DIR = .
DOCKERFILE = ${DOCKER_FILE_DIR}/Dockerfile
NUM_BUILD_CORES ?= 1
IMAGE_NAME = ${APP_NAME}
OUTPUT_DIR ?= $(shell pwd)/outputs

# Handle Optional GPU
ifeq ($(USE_GPU),true)
	DOCKER_GPU_ARG = --gpus all
endif

DOCKER_CORE_ARGS = \
	--env XPASSTHROUGH=$(XPASSTHROUGH) \
	--env DISPLAY=$(DISPLAY) \
	--volume="$(OUTPUT_DIR):/outputs/:rw" \
	$(DOCKER_GPU_ARG)
	

DOCKER_DEVEL_VOLUMES = \
	--volume="$(shell pwd)/scripts:/scripts:rw"

DOCKER_PYTHON = @docker run --rm --init --net=host \
		$(DOCKER_ARGS) $(DOCKER_CORE_ARGS) \
		${IMAGE_NAME}:${VERSION} python3 \


.PHONY: help
help:
	@echo ''
	@echo 'Usage: make [TARGET] [EXTRA_ARGUMENTS]'
	@echo 'Core Targets:'
	@echo '  help		display this help message'
	@echo '  build		build docker image (incremental)'
	@echo '  rebuild	build docker image from scratch'
	@echo '  kill		close all project-related docker containers'
	@echo '  term		open a terminal in the docker container'
	@echo '  devel		term, but with local code folders mounted'
	@echo 'Demo Targets:'
	@echo '  all 		(or "all-demos") runs all demos'
	@echo '  run Complex-yolov4'

# === Helper functions ===

define xhost_activate
	@echo "Enabling local xhost sharing:"
	@echo "  Display: $(DISPLAY)"
	@-DISPLAY=$(DISPLAY) xhost  +
	@-xhost +
endef

.PHONY: kill
kill:
	@echo "Closing all running docker containers:"
	@docker kill $(shell docker ps -q --filter ancestor=${IMAGE_NAME}:${VERSION})

# === Build Targets ===
.PHONY: build
build:
	@-mkdir $(RESULTS_DIR)
	@docker build -t ${IMAGE_NAME}:${VERSION} \
		--build-arg NUM_BUILD_CORES=$(NUM_BUILD_CORES) \
		$(DOCKER_ARGS) -f ./${DOCKERFILE} .

.PHONY: rebuild
rebuild:
	@docker build -t ${IMAGE_NAME}:${VERSION} --no-cache \
		--build-arg NUM_BUILD_CORES=$(NUM_BUILD_CORES) \
		$(DOCKER_ARGS) -f ./${DOCKERFILE} .

# ===== Development targets =====

.PHONY: term devel
term:
	@docker run -it --init $(DOCKER_GPU) --net=host \
		--gpus=device=0 \
		$(DOCKER_ARGS) $(DOCKER_CORE_ARGS) \
		${IMAGE_NAME}:${VERSION} /bin/bash
devel:
	@docker run -it --init --net=host \
		$(DOCKER_ARGS) $(DOCKER_CORE_ARGS) $(DOCKER_DEVEL_VOLUMES)\
		${IMAGE_NAME}:${VERSION} /bin/bash

# test: build
# 	$(DOCKER_PYTHON) -m py.test \
# 		-rsx \
# 		--unity_exe_path /unity/$(UNITY_DBG_BASENAME).x86_64 \
# 		tests


# ===== Demo scripts =====

# Run example
.PHONY: create-a-image
create-a-image:
	@$(call xhost-activate)
	@echo "Demo: Running the complex-yolov4 network"
	@$(DOCKER_PYTHON) -m scripts.check \
		--output_image /outputs/demo_plotting.png


# Run Traffic-Sign Recognition Models
.PHONY: training-traffic-sign-reg eval-traffic-sign-reg
training-traffic-sign-reg:
	@echo "Training the traffic-sign-reg network"
	@docker run --init --net=host \
		--gpus=device=0 \
		$(DOCKER_ARGS) $(DOCKER_CORE_ARGS) \
		${IMAGE_NAME}:${VERSION} \
		python3 -m scripts.regconition1.train 

eval-traffic-sign-reg: XPASSTHROUGH=true
eval-traffic-sign-reg:
	@echo "Evaluate the traffic-sign-reg network"
	@docker run --init --rm --net=host \
		--gpus=device=0 \
		$(DOCKER_ARGS) $(DOCKER_CORE_ARGS) \
		${IMAGE_NAME}:${VERSION} \
		python3 -m scripts.regconition1.evaluate \
		--model scripts/regconition1/traffic_models/model_1.pt \
		--xpassthrough $(XPASSTHROUGH)

# Run Alexnet for traffic sign regconition - GPU
.PHONY: test-alexnet-trafffsign-reg 
test-alexnet-trafffsign-reg:
	@echo "Evaluate the alexnet-trafffsign-reg network"
	@docker run --rm --init --net=host \
		--gpus=device=0 \
		$(DOCKER_ARGS) $(DOCKER_CORE_ARGS) \
		${IMAGE_NAME}:${VERSION} \
		python3 -m scripts.regconition2.traffic_sign_recognition


# Run Yolo-v4 - Point Cloud data  GPU
.PHONY: test-yolov4-pcl-vehicle-detection 
test-yolov4-pcl-vehicle-detection :
	@echo "test yolov4 pcl vehicles detection network"
	@docker run --rm --init --net=host \
		--gpus=device=0 \
		$(DOCKER_ARGS) $(DOCKER_CORE_ARGS) \
		${IMAGE_NAME}:${VERSION} \
		python3 -m scripts.pcl.test \
		--cfgfile scripts/pcl/config/cfg/complex_yolov4.cfg\
		--pretrained_path scripts/pcl/complex-yolov4/complex_yolov4_mse_loss.pth\
		--dataPath scripts/pcl/dataset/kitti\
		--gpu_idx 0\
		--num_samples 5 \
		--save_test_output 




# === Plotting ===
# A target that plots to the screen
.PHONY: image-visualize
image-visualize: XPASSTHROUGH=true
image-visualize:
	@echo "Demo: Plotting from within Docker"
	@$(DOCKER_PYTHON) -m scripts.check \
		--xpassthrough $(XPASSTHROUGH)

.PHONY: all-demos
all-demos: demo-plotting
	@echo "Completed all demos successfully."
