set(CPU_MLIR ${CINDER_BINDING_PATH}/fcg_export_cpu.mlir)
set(GPU_MLIR ${CINDER_BINDING_PATH}/fcg_export_gpu.mlir)

set(BASELINE_OUTPUT libFCGPassBaseline.so)
set(BASELINE_OUTPUT_script
    $<TARGET_FILE:cinder-opt> --fcg-pass-fuse-full-flow --fcg-pass-baseline ${CPU_MLIR} > ${BASELINE_OUTPUT}.mlir && 
    ${MLIR_OPT} --pass-pipeline='builtin.module(
        convert-scf-to-cf,
        convert-cf-to-llvm,
        convert-arith-to-llvm,
        finalize-memref-to-llvm,
        convert-func-to-llvm,
        reconcile-unrealized-casts
    )' ${BASELINE_OUTPUT}.mlir > ${BASELINE_OUTPUT}.llvm.mlir &&
    ${MLIR_TRANSLATE} --mlir-to-llvmir ${BASELINE_OUTPUT}.llvm.mlir > ${BASELINE_OUTPUT}.llvm.ll && 
    ${CLANG} -shared -fPIC -O2 -march=native -lm -mavx2 -Wno-override-module ${BASELINE_OUTPUT}.llvm.ll -o ${BASELINE_OUTPUT}
)
add_custom_command(
    OUTPUT ${BASELINE_OUTPUT}
    DEPENDS ${CPU_MLIR} cinder-opt
    COMMAND bash -c "\"" ${BASELINE_OUTPUT_script} "\""
    COMMENT "Generating ${BASELINE_OUTPUT}"
)

set(VECTORIZE_OUTPUT libFCGPassVectorize.so)
set(VECTORIZE_OUTPUT_script
    $<TARGET_FILE:cinder-opt> --fcg-pass-fuse-full-flow --fcg-pass-vectorize ${CPU_MLIR} > ${VECTORIZE_OUTPUT}.mlir &&
    ${MLIR_OPT} --pass-pipeline='builtin.module(
        convert-scf-to-cf,
        convert-vector-to-llvm,
        convert-cf-to-llvm,
        convert-arith-to-llvm,
        finalize-memref-to-llvm,
        convert-func-to-llvm,
        reconcile-unrealized-casts
    )' ${VECTORIZE_OUTPUT}.mlir > ${VECTORIZE_OUTPUT}.llvm.mlir &&
    ${MLIR_TRANSLATE} --mlir-to-llvmir ${VECTORIZE_OUTPUT}.llvm.mlir > ${VECTORIZE_OUTPUT}.llvm.ll &&
    ${CLANG} -shared -fPIC -O2 -march=native -lm -mavx2 -Wno-override-module ${VECTORIZE_OUTPUT}.llvm.ll -o ${VECTORIZE_OUTPUT}
)
add_custom_command(
    OUTPUT ${VECTORIZE_OUTPUT}
    DEPENDS ${CPU_MLIR} cinder-opt
    COMMAND bash -c "\"" ${VECTORIZE_OUTPUT_script} "\""
    COMMENT "Generating ${VECTORIZE_OUTPUT}"
)

set(LINALG_OUTPUT libFCGPassLinalg.so)
set(LINALG_OUTPUT_script
    $<TARGET_FILE:cinder-opt> --fcg-pass-fuse-full-flow --fcg-pass-linalg ${CPU_MLIR} > ${LINALG_OUTPUT}.mlir &&
    ${MLIR_OPT} --pass-pipeline='builtin.module(
        reconcile-unrealized-casts,
        func.func(
            convert-linalg-to-affine-loops,
            affine-loop-fusion{mode=greedy},
            affine-loop-invariant-code-motion,
            affine-loop-tile{tile-sizes=128,128},
            affine-loop-unroll{unroll-factor=8},
            affine-super-vectorize{virtual-vector-size=32 vectorize-reductions},
            lower-affine,
            convert-scf-to-cf
        ),
        convert-vector-to-llvm,
        convert-cf-to-llvm,
        convert-arith-to-llvm,
        finalize-memref-to-llvm,
        convert-func-to-llvm,
        reconcile-unrealized-casts
    )' ${LINALG_OUTPUT}.mlir > ${LINALG_OUTPUT}.llvm.mlir &&
    ${MLIR_TRANSLATE} --mlir-to-llvmir ${LINALG_OUTPUT}.llvm.mlir > ${LINALG_OUTPUT}.llvm.ll &&
    ${CLANG} -shared -fPIC -O2 -march=native -lm -mavx2 -Wno-override-module ${LINALG_OUTPUT}.llvm.ll -o ${LINALG_OUTPUT}
)
add_custom_command(
    OUTPUT ${LINALG_OUTPUT}
    DEPENDS ${CPU_MLIR} cinder-opt
    COMMAND bash -c "\"" ${LINALG_OUTPUT_script} "\""
    COMMENT "Generating ${LINALG_OUTPUT}"
)

set(GPU_OUTPUT libFCGPassGPU.so)
set(GPU_OUTPUT_script
    $<TARGET_FILE:cinder-opt> --fcg-pass-fuse-full-flow --fcg-pass-gpu ${GPU_MLIR} > ${GPU_OUTPUT}.mlir &&
    ${MLIR_OPT} --pass-pipeline='builtin.module(
        convert-scf-to-cf,
        nvvm-attach-target{chip=${CINDER_CUDA_SMLEVEL} O=3},
        gpu.module(convert-gpu-to-nvvm),
        gpu-to-llvm,gpu-module-to-binary,
        convert-nvvm-to-llvm,
        convert-index-to-llvm,
        convert-arith-to-llvm,
        convert-cf-to-llvm,
        convert-func-to-llvm,
        finalize-memref-to-llvm,
        canonicalize,
        cse,
        reconcile-unrealized-casts
    )' ${GPU_OUTPUT}.mlir > ${GPU_OUTPUT}.llvm.mlir &&
    ${MLIR_TRANSLATE} --mlir-to-llvmir ${GPU_OUTPUT}.llvm.mlir > ${GPU_OUTPUT}.llvm.ll &&
    ${CLANG} -shared -fPIC -O2 -Wno-override-module -lm ${GPU_OUTPUT}.llvm.ll -o ${GPU_OUTPUT}
        -mcpu=${CINDER_CUDA_SMLEVEL}
        ${CINDER_CUDA_LINKAGE}
        -Wl,-rpath,${MLIR_TOOL_DIR}/lib
)
add_custom_command(
    OUTPUT ${GPU_OUTPUT}
    DEPENDS ${GPU_MLIR} cinder-opt
    COMMAND bash -c "\"" ${GPU_OUTPUT_script} "\""
    COMMENT "Generating ${GPU_OUTPUT}"
)

set(NVVM_OUTPUT libFCGPassNVVM.so)
set(NVVM_OUTPUT_script
    $<TARGET_FILE:cinder-opt> --fcg-pass-fuse-full-flow ${GPU_MLIR} > ${NVVM_OUTPUT}-fuse-full-flow.mlir &&
    $<TARGET_FILE:cinder-opt> --fcg-pass-nvvm-gpu ${NVVM_OUTPUT}-fuse-full-flow.mlir > ${NVVM_OUTPUT}.gpu.mlir &&
    ${MLIR_OPT} --pass-pipeline='builtin.module(
        convert-scf-to-cf,
        convert-cf-to-llvm,
        convert-arith-to-llvm,
        reconcile-unrealized-casts
    )' ${NVVM_OUTPUT}.gpu.mlir > ${NVVM_OUTPUT}.gpu.llvm.mlir &&
    ${MLIR_TRANSLATE} --mlir-to-llvmir ${NVVM_OUTPUT}.gpu.llvm.mlir -o ${NVVM_OUTPUT}.gpu.llvm.ll &&
    ${LLC} -march=nvptx64 -mcpu=${CINDER_CUDA_SMLEVEL} -filetype=asm ${NVVM_OUTPUT}.gpu.llvm.ll -o ${NVVM_OUTPUT}.gpu.ptx &&
    ${PTXAS} -arch=${CINDER_CUDA_SMLEVEL} ${NVVM_OUTPUT}.gpu.ptx -o ${NVVM_OUTPUT}.gpu.cubin &&

    $<TARGET_FILE:cinder-opt> --fcg-pass-nvvm-cpu=kernel-path=${NVVM_OUTPUT}.gpu.cubin ${NVVM_OUTPUT}-fuse-full-flow.mlir > ${NVVM_OUTPUT}.cpu.mlir &&
    ${MLIR_OPT} --pass-pipeline='builtin.module(
        finalize-memref-to-llvm,
        convert-func-to-llvm,
        convert-arith-to-llvm,
        convert-scf-to-cf,
        convert-cf-to-llvm,
        reconcile-unrealized-casts
    )' ${NVVM_OUTPUT}.cpu.mlir > ${NVVM_OUTPUT}.cpu.llvm.mlir &&
    ${MLIR_TRANSLATE} --mlir-to-llvmir ${NVVM_OUTPUT}.cpu.llvm.mlir -o ${NVVM_OUTPUT}.cpu.llvm.ll &&
    ${LLC} -filetype=obj -relocation-model=pic ${NVVM_OUTPUT}.cpu.llvm.ll -o ${NVVM_OUTPUT}.cpu_with_cubin.o &&
    ${GCC} -shared -fPIC -lm -o ${NVVM_OUTPUT} ${NVVM_OUTPUT}.cpu_with_cubin.o
        ${CINDER_CUDA_LINKAGE}
)
add_custom_command(
    OUTPUT ${NVVM_OUTPUT}
    DEPENDS ${GPU_MLIR} cinder-opt
    COMMAND bash -c "\"" ${NVVM_OUTPUT_script} "\""
    COMMENT "Generating ${NVVM_OUTPUT}"
)

set(ALL_LIBS
    ${BASELINE_OUTPUT}
    ${VECTORIZE_OUTPUT}
    ${LINALG_OUTPUT}
    ${GPU_OUTPUT}
    ${NVVM_OUTPUT}
)

add_custom_target(fcg-pass-lib ALL DEPENDS ${ALL_LIBS})
list(TRANSFORM ALL_LIBS PREPEND "${CMAKE_CURRENT_BINARY_DIR}/")
install(FILES ${ALL_LIBS} DESTINATION ${CINDER_LIBRARY_PATH})
