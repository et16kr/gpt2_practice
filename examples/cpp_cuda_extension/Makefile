PYTHON ?= python3

.PHONY: help build run run-cpu clean

help:
	@echo "Targets:"
	@echo "  make build    - build/load extension once (no tensor demo)"
	@echo "  make run      - build (if needed) and run demo"
	@echo "  make run-cpu  - force CPU path and run demo"
	@echo "  make clean    - remove local build artifacts"

build:
	$(PYTHON) -c "import demo; demo.build_extension(); print('extension built')"

run:
	$(PYTHON) demo.py

run-cpu:
	CUDA_VISIBLE_DEVICES='' $(PYTHON) demo.py

clean:
	rm -rf build
