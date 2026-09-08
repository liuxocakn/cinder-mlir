LLVM_COMMIT  		:= 26eb4285b56edd8c897642078d91f16ff0fd3472
LLVM_SRC_DIR 		:= ./llvm-project
LLVM_OBJ_DIR		:= ./llvm-mlir-bin
LLVM_BUILD_DIR      := ./build-llvm
LLVM_MLIR_PACKAGE   := ${LLVM_OBJ_DIR}/lib/cmake/mlir

LLVM_COMMIT_CHECK:
	cd ${LLVM_SRC_DIR} && git checkout ${LLVM_COMMIT} && cd -

${LLVM_OBJ_DIR}: LLVM_COMMIT_CHECK
	mkdir -p ${LLVM_BUILD_DIR} && cd ${LLVM_BUILD_DIR} && \
	cmake ../${LLVM_SRC_DIR}/llvm \
		-G Ninja \
		-DCMAKE_INSTALL_PREFIX=../${LLVM_OBJ_DIR} \
		-DLLVM_ENABLE_PROJECTS="mlir;llvm" \
		-DLLVM_BUILD_EXAMPLES=ON \
		-DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_CXX_FLAGS_DEBUG="-O0 -g3" \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DMLIR_ENABLE_CUDA_RUNNER=ON \
		-DCUDAToolkit_ROOT=/usr/local/cuda \
		-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
		-DCMAKE_C_COMPILER=/usr/bin/gcc-10 \
		-DCMAKE_CXX_COMPILER=/usr/bin/g++-10 \
		&& \
	ninja install

llvm-mlir-install: ${LLVM_OBJ_DIR}
llvm-mlir-clean:
	rm -rf ${LLVM_OBJ_DIR} ${LLVM_BUILD_DIR}
