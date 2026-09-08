function(add_ctest target_name source_files _cflags _include _link _depend)
    add_executable(${target_name} ${source_files})
    add_dependencies(${target_name} ${_depend})
    target_include_directories(${target_name} PRIVATE ${_include})
    target_compile_options(${target_name} PRIVATE ${_cflags})
    target_link_directories(${target_name} PRIVATE ${CINDER_LIBRARY_PATH})
    target_link_directories(${target_name} PRIVATE ${CMAKE_BINARY_DIR}/binding)
    set_target_properties(${target_name} PROPERTIES
        BUILD_RPATH "${CMAKE_BINARY_DIR}/binding"
        INSTALL_RPATH "\$ORIGIN/../lib"
        OUTPUT_NAME ${target_name}
    )
    target_link_libraries(${target_name} PRIVATE ${_link})
    install(TARGETS ${target_name} RUNTIME DESTINATION ${CINDER_INSTALL_PATH})
endfunction()

add_ctest(fcg-pass-baseline
	"mainCPU.c"
	"-O3"
	"${CINDER_BINDING_PATH}/c/include"
	"m;FCGPassBaseline"
    "fcg-pass-lib"
)

add_ctest(fcg-pass-vectorize
    "mainCPU.c"
    "-O3"
    "${CINDER_BINDING_PATH}/c/include"
    "m;FCGPassVectorize"
    "fcg-pass-lib"
)

add_ctest(fcg-pass-linalg
    "mainCPU.c"
    "-O3"
    "${CINDER_BINDING_PATH}/c/include"
    "m;FCGPassLinalg"
    "fcg-pass-lib"
)

set(GPU_LINKAGE
    "m"
    ${CINDER_CUDA_LINK_FLAGS}
)

set(GPU_INCLUDE
    "${CINDER_BINDING_PATH}/c/include"
    ${CINDER_CUDA_INCLUDE}
)

add_ctest(fcg-pass-gpu
    "mainGPU.c"
    "-O3"
    "${GPU_INCLUDE}"
    "${GPU_LINKAGE};FCGPassGPU"
    "fcg-pass-lib"
)

add_ctest(fcg-pass-nvvm
    "mainGPU.c"
    "-O3"
    "${GPU_INCLUDE}"
    "${GPU_LINKAGE};FCGPassNVVM"
    "fcg-pass-lib"
)
