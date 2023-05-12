MAJOR ?= 0
MINOR ?= 1
VERSION = $(MAJOR).$(MINOR)
APP_NAME ?= ccs695-ubuntu20

# Docker args
DISPLAY ?= :0.0
XPASSTHROUGH ?= true
DOCKER_FILE_DIR = .
DOCKERFILE = ${DOCKER_FILE_DIR}/Dockerfile
NUM_BUILD_CORES ?= 1
IMAGE_NAME = ${APP_NAME}
DATA_BASE_DIR ?= $(shell pwd)/data
MODEL_BASE_DIR ?= $(shell pwd)/models
OUTPUT_BASE_DIR ?= $(shell pwd)/outputs
CODE_BASE_DIR ?= $(shell pwd)/scripts

# Handle Optional GPU
ifeq ($(USE_GPU),true)
	DOCKER_GPU_ARG = --gpus all
endif

DOCKER_CORE_ARGS = \
	--env XPASSTHROUGH=$(XPASSTHROUGH) \
	--env DISPLAY=$(DISPLAY) \
	--volume="$(OUTPUT_BASE_DIR):/outputs:rw" \
	$(DOCKER_GPU_ARG)

DOCKER_DEVEL_VOLUMES = \
	--volume="$(shell pwd)/requirements.txt:/requirements.txt:rw"

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
	@echo '  demo-batch-parallel (run with "-j6") shows off parallel job running with make'
	@echo '  demo-plotting  plots via matplotlib inside the container'
	@echo '  demo-pybind	runs C++ code in python via pybind'
	@echo '  demo-unity-env runs Unity in headless mode, generates plot'

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
	@-mkdir $(DATA_BASE_DIR)
	@docker build -t ${IMAGE_NAME}:${VERSION} \
		--build-arg NUM_BUILD_CORES=$(NUM_BUILD_CORES) \
		$(DOCKER_ARGS) -f ./${DOCKERFILE} .

.PHONY: rebuild
rebuild:
	@docker build -t ${IMAGE_NAME}:${VERSION} --no-cache \
		--build-arg NUM_BUILD_CORES=$(NUM_BUILD_CORES) \
		$(DOCKER_ARGS) -f ./${DOCKERFILE} .

# ===== Development targets =====

.PHONY: term devel test
term:
	@docker run -it --init $(DOCKER_GPU) --net=host \
		--gpus=device=0 \
		$(DOCKER_ARGS) $(DOCKER_CORE_ARGS) \
		${IMAGE_NAME}:${VERSION} /bin/bash
devel:
	@docker run -it --init --net=host \
		$(DOCKER_ARGS) $(DOCKER_CORE_ARGS) $(DOCKER_DEVEL_VOLUMES)\
		${IMAGE_NAME}:${VERSION} /bin/bash


# ===== Demo scripts =====


# === Plotting ===

# === Running CNN models =======
# A target that runs the tensorflow model for traffic sign detection
.PHONY: test-traffic-sign-detection test-traffic-light-detection 
test-traffic-sign-detection: XPASSTHROUGH=true
test-traffic-sign-detection:
	@echo "Test the network: traffic-sign-detection"
	@docker run --init --net=host \
		--gpus=device=0 \
		$(DOCKER_ARGS) $(DOCKER_CORE_ARGS) \
		${IMAGE_NAME}:${VERSION} \
		python3 -m scripts.ts_detection.traffic_sign_detection \
		--xpassthrough $(XPASSTHROUGH)


test-traffic-light-detection : XPASSTHROUGH=true
test-traffic-light-detection :
	@echo "Test the network: test-traffic-light-detection "
	@docker run --init --net=host \
		--gpus=device=0 \
		$(DOCKER_ARGS) $(DOCKER_CORE_ARGS) \
		${IMAGE_NAME}:${VERSION} \
		python3 -m scripts.tl_detection.main \
		--xpassthrough $(XPASSTHROUGH)