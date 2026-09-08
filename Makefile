all: cinder

include scripts/llvm-mlir-install.mk

cinder:
	cmake -S . -B build -G Ninja -Wno-dev -DMLIR_TOOL_DIR=./llvm-mlir-bin
	cmake --build build -j$(nproc)
	cmake --install build

tree:
	tree -I "build*|llvm-*" .

test:
	@echo ==========================================================================
	@echo "                               C binding                                "
	@echo
	@./test/bin/fcg-pass-baseline
	@./test/bin/fcg-pass-vectorize
	@./test/bin/fcg-pass-linalg
	@./test/bin/fcg-pass-gpu
	@./test/bin/fcg-pass-nvvm
	@echo ==========================================================================
	@echo "                              Python binding                            "
	@echo
	@python3 ./test/python/mainCPU.py --pass=Baseline
	@python3 ./test/python/mainCPU.py --pass=Vectorize
	@python3 ./test/python/mainCPU.py --pass=Linalg
	@python3 ./test/python/mainGPU.py --pass=GPU
	@python3 ./test/python/mainGPU.py --pass=NVVM
	@echo ==========================================================================

clean:
	rm -rf test/bin test/lib

clean-build: clean
	rm -rf build/

clean-all: clean-build llvm-mlir-clean

.PHONY: test clean clean-build clean-all
