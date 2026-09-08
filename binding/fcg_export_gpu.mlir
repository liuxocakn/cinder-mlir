module {

    llvm.func @cudaMalloc(%ptr: !llvm.ptr<!llvm.ptr>, %size: i64) -> i32
    llvm.func @cudaFree(%ptr: !llvm.ptr) -> i32
    llvm.func @cudaMemcpy(%dst: !llvm.ptr, %src: !llvm.ptr, %count: i64, %kind: i32) -> i32

    func.func @test_matmul_f32(
        %arg0: tensor<?x?xf32>,
        %arg1: tensor<?x?xf32>,
        %out0: tensor<?x?xf32>
    ) -> tensor<?x?xf32> {
        %0 = fcg.matmul %arg0, %arg1, %out0
            : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
        func.return %0 : tensor<?x?xf32>
    }

    func.func @test_matmul_f16(
        %arg0: tensor<?x?xf16>,
        %arg1: tensor<?x?xf16>,
        %out0: tensor<?x?xf16>
    ) -> tensor<?x?xf16> {
        %0 = fcg.matmul %arg0, %arg1, %out0
            : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16> -> tensor<?x?xf16>
        func.return %0 : tensor<?x?xf16>
    }

    func.func @test_add_f32(
        %arg0: tensor<?x?xf32>,
        %arg1: tensor<?x?xf32>,
        %out0: tensor<?x?xf32>
    ) -> tensor<?x?xf32> {
        %0 = fcg.add %arg0, %arg1, %out0
            : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
        func.return %0 : tensor<?x?xf32>
    }

    func.func @test_add_f16(
        %arg0: tensor<?x?xf16>,
        %arg1: tensor<?x?xf16>,
        %out0: tensor<?x?xf16>
    ) -> tensor<?x?xf16> {
        %0 = fcg.add %arg0, %arg1, %out0
            : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16> -> tensor<?x?xf16>
        func.return %0 : tensor<?x?xf16>
    }

    func.func @test_relu_f32(
        %arg0: tensor<?x?xf32>,
        %out0: tensor<?x?xf32>
    ) -> tensor<?x?xf32> {
        %0 = fcg.relu %arg0, %out0
            : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
        func.return %0 : tensor<?x?xf32>
    }

    func.func @test_relu_f16(
        %arg0: tensor<?x?xf16>,
        %out0: tensor<?x?xf16>
    ) -> tensor<?x?xf16> {
        %0 = fcg.relu %arg0, %out0
            : tensor<?x?xf16>, tensor<?x?xf16> -> tensor<?x?xf16>
        func.return %0 : tensor<?x?xf16>
    }

    func.func @test_full_flow_f32(
        %A: tensor<?x?xf32>,
        %B: tensor<?x?xf32>,
        %bias: tensor<?x?xf32>,
        %out0: tensor<?x?xf32>
    ) -> tensor<?x?xf32> {
        %0 = fcg.matmul %A, %B, %out0
            : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
        %1 = fcg.add %0, %bias, %out0
            : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
        %2 = fcg.relu %1, %out0
            : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
        func.return %2 : tensor<?x?xf32>
    }

    func.func @test_full_flow_f16(
        %A: tensor<?x?xf16>,
        %B: tensor<?x?xf16>,
        %bias: tensor<?x?xf16>,
        %out0: tensor<?x?xf16>
    ) -> tensor<?x?xf16> {
        %0 = fcg.matmul %A, %B, %out0
            : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16> -> tensor<?x?xf16>
        %1 = fcg.add %0, %bias, %out0
            : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16> -> tensor<?x?xf16>
        %2 = fcg.relu %1, %out0
            : tensor<?x?xf16>, tensor<?x?xf16> -> tensor<?x?xf16>
        func.return %2 : tensor<?x?xf16>
    }

    func.func @test_fusion_full_flow_f32(
        %A: tensor<?x?xf32>,
        %B: tensor<?x?xf32>,
        %bias: tensor<?x?xf32>,
        %out0: tensor<?x?xf32>
    ) -> tensor<?x?xf32> {
        %0 = fcg.matmul %A, %B, %out0
            : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
        %1 = fcg.add %0, %bias, %out0
            : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
        %2 = fcg.relu %1, %out0
            : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
        func.return %2 : tensor<?x?xf32>
    }

    func.func @test_fusion_full_flow_f16(
        %A: tensor<?x?xf16>,
        %B: tensor<?x?xf16>,
        %bias: tensor<?x?xf16>,
        %out0: tensor<?x?xf16>
    ) -> tensor<?x?xf16> {
        %0 = fcg.matmul %A, %B, %out0
            : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16> -> tensor<?x?xf16>
        %1 = fcg.add %0, %bias, %out0
            : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16> -> tensor<?x?xf16>
        %2 = fcg.relu %1, %out0
            : tensor<?x?xf16>, tensor<?x?xf16> -> tensor<?x?xf16>
        func.return %2 : tensor<?x?xf16>
    }

    func.func @test_memref_alloc_f32(
        %x: index,
        %y: index
    ) -> memref<?x?xf32> {
        %alloc = memref.alloc(%x, %y) {alignment = 64 : i64} : memref<?x?xf32>
        return %alloc : memref<?x?xf32>
    }

    func.func @test_memref_free_f32(%memref: memref<?x?xf32>) -> () {
        memref.dealloc %memref : memref<?x?xf32>
        return
    }

    func.func @test_memref_alloc_f16(
        %x: index,
        %y: index
    ) -> memref<?x?xf16> {
        %alloc = memref.alloc(%x, %y) {alignment = 64 : i64} : memref<?x?xf16>
        return %alloc : memref<?x?xf16>
    }

    func.func @test_memref_free_f16(%memref: memref<?x?xf16>) -> () {
        memref.dealloc %memref : memref<?x?xf16>
        return
    }

    func.func @test_gpu_memref_alloc_f32(
        %x: index,
        %y: index
    ) -> memref<?x?xf32, 1> {
        %num_elem = arith.muli %x, %y : index
        %elem_size = arith.constant 4 : index
        %bytes = arith.muli %num_elem, %elem_size : index
        %bytes_i64 = arith.index_cast %bytes : index to i64

        %one_i32 = arith.constant 1 : i32
        %ptr_var = llvm.alloca %one_i32 x !llvm.ptr : (i32) -> !llvm.ptr<!llvm.ptr>
        %err = llvm.call @cudaMalloc(%ptr_var, %bytes_i64) : (!llvm.ptr<!llvm.ptr>, i64) -> i32

        %dev_ptr = llvm.load %ptr_var : !llvm.ptr<!llvm.ptr>
        %dev_ptr3 = llvm.addrspacecast %dev_ptr : !llvm.ptr to !llvm.ptr<1>

        %desc = llvm.mlir.undef : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %desc1 = llvm.insertvalue %dev_ptr3, %desc[0] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %desc2 = llvm.insertvalue %dev_ptr3, %desc1[1] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %c0_i64 = arith.constant 0 : i64
        %desc3 = llvm.insertvalue %c0_i64, %desc2[2] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %x_i64 = arith.index_cast %x : index to i64
        %y_i64 = arith.index_cast %y : index to i64
        %desc4 = llvm.insertvalue %x_i64, %desc3[3, 0] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %desc5 = llvm.insertvalue %y_i64, %desc4[3, 1] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %desc6 = llvm.insertvalue %y_i64, %desc5[4, 0] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %c1_i64 = arith.constant 1 : i64
        %desc7 = llvm.insertvalue %c1_i64, %desc6[4, 1] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>

        %memref = builtin.unrealized_conversion_cast %desc7 : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)> to memref<?x?xf32, 1>
        return %memref : memref<?x?xf32, 1>
    }

    func.func @test_gpu_memref_free_f32(%memref: memref<?x?xf32, 1>) -> () {
        %desc = builtin.unrealized_conversion_cast %memref : memref<?x?xf32, 1> to !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %dev_ptr3 = llvm.extractvalue %desc[0] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %dev_ptr = llvm.addrspacecast %dev_ptr3 : !llvm.ptr<1> to !llvm.ptr
        %err = llvm.call @cudaFree(%dev_ptr) : (!llvm.ptr) -> i32
        return
    }

    func.func @test_gpu_memref_alloc_f16(
        %x: index,
        %y: index
    ) -> memref<?x?xf16, 1> {
        %num_elem = arith.muli %x, %y : index
        %elem_size = arith.constant 2 : index
        %bytes = arith.muli %num_elem, %elem_size : index
        %bytes_i64 = arith.index_cast %bytes : index to i64

        %one_i32 = arith.constant 1 : i32
        %ptr_var = llvm.alloca %one_i32 x !llvm.ptr : (i32) -> !llvm.ptr<!llvm.ptr>
        %err = llvm.call @cudaMalloc(%ptr_var, %bytes_i64) : (!llvm.ptr<!llvm.ptr>, i64) -> i32

        %dev_ptr = llvm.load %ptr_var : !llvm.ptr<!llvm.ptr>
        %dev_ptr3 = llvm.addrspacecast %dev_ptr : !llvm.ptr to !llvm.ptr<1>

        %desc = llvm.mlir.undef : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %desc1 = llvm.insertvalue %dev_ptr3, %desc[0] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %desc2 = llvm.insertvalue %dev_ptr3, %desc1[1] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %c0_i64 = arith.constant 0 : i64
        %desc3 = llvm.insertvalue %c0_i64, %desc2[2] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %x_i64 = arith.index_cast %x : index to i64
        %y_i64 = arith.index_cast %y : index to i64
        %desc4 = llvm.insertvalue %x_i64, %desc3[3, 0] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %desc5 = llvm.insertvalue %y_i64, %desc4[3, 1] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %desc6 = llvm.insertvalue %y_i64, %desc5[4, 0] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %c1_i64 = arith.constant 1 : i64
        %desc7 = llvm.insertvalue %c1_i64, %desc6[4, 1] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>

        %memref = builtin.unrealized_conversion_cast %desc7 : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)> to memref<?x?xf16, 1>
        return %memref : memref<?x?xf16, 1>
    }

    func.func @test_gpu_memref_free_f16(%memref: memref<?x?xf16, 1>) -> () {
        %desc = builtin.unrealized_conversion_cast %memref : memref<?x?xf16, 1> to !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %dev_ptr3 = llvm.extractvalue %desc[0] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %dev_ptr = llvm.addrspacecast %dev_ptr3 : !llvm.ptr<1> to !llvm.ptr
        %err = llvm.call @cudaFree(%dev_ptr) : (!llvm.ptr) -> i32
        return
    }

    func.func @test_gpu_memref_copy_h2d_f32(
        %src: memref<?x?xf32>,
        %dst: memref<?x?xf32, 1>
    ) -> () {
        %host_desc = builtin.unrealized_conversion_cast %src : memref<?x?xf32> to !llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %host_ptr = llvm.extractvalue %host_desc[1] : !llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>

        %dev_desc = builtin.unrealized_conversion_cast %dst : memref<?x?xf32, 1> to !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %dev_ptr3 = llvm.extractvalue %dev_desc[1] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %dev_ptr = llvm.addrspacecast %dev_ptr3 : !llvm.ptr<1> to !llvm.ptr

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %m = memref.dim %dst, %c0 : memref<?x?xf32, 1>
        %n = memref.dim %dst, %c1 : memref<?x?xf32, 1>
        %num_elem = arith.muli %m, %n : index
        %elem_size = arith.constant 4 : index
        %bytes = arith.muli %num_elem, %elem_size : index
        %bytes_i64 = arith.index_cast %bytes : index to i64

        %kind = arith.constant 1 : i32
        %err = llvm.call @cudaMemcpy(%dev_ptr, %host_ptr, %bytes_i64, %kind) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
        return
    }

    func.func @test_gpu_memref_copy_d2h_f32(
        %src: memref<?x?xf32, 1>,
        %dst: memref<?x?xf32>
    ) -> () {
        %host_desc = builtin.unrealized_conversion_cast %dst : memref<?x?xf32> to !llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %host_ptr = llvm.extractvalue %host_desc[1] : !llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>

        %dev_desc = builtin.unrealized_conversion_cast %src : memref<?x?xf32, 1> to !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %dev_ptr3 = llvm.extractvalue %dev_desc[1] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %dev_ptr = llvm.addrspacecast %dev_ptr3 : !llvm.ptr<1> to !llvm.ptr

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %m = memref.dim %src, %c0 : memref<?x?xf32, 1>
        %n = memref.dim %src, %c1 : memref<?x?xf32, 1>
        %num_elem = arith.muli %m, %n : index
        %elem_size = arith.constant 4 : index
        %bytes = arith.muli %num_elem, %elem_size : index
        %bytes_i64 = arith.index_cast %bytes : index to i64

        %kind = arith.constant 2 : i32
        %err = llvm.call @cudaMemcpy(%host_ptr, %dev_ptr, %bytes_i64, %kind) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
        return
    }

    func.func @test_gpu_memref_copy_h2d_f16(
        %src: memref<?x?xf16>,
        %dst: memref<?x?xf16, 1>
    ) -> () {
        %host_desc = builtin.unrealized_conversion_cast %src : memref<?x?xf16> to !llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %host_ptr = llvm.extractvalue %host_desc[1] : !llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>

        %dev_desc = builtin.unrealized_conversion_cast %dst : memref<?x?xf16, 1> to !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %dev_ptr3 = llvm.extractvalue %dev_desc[1] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %dev_ptr = llvm.addrspacecast %dev_ptr3 : !llvm.ptr<1> to !llvm.ptr

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %m = memref.dim %dst, %c0 : memref<?x?xf16, 1>
        %n = memref.dim %dst, %c1 : memref<?x?xf16, 1>
        %num_elem = arith.muli %m, %n : index
        %elem_size = arith.constant 2 : index
        %bytes = arith.muli %num_elem, %elem_size : index
        %bytes_i64 = arith.index_cast %bytes : index to i64

        %kind = arith.constant 1 : i32
        %err = llvm.call @cudaMemcpy(%dev_ptr, %host_ptr, %bytes_i64, %kind) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
        return
    }

    func.func @test_gpu_memref_copy_d2h_f16(
        %src: memref<?x?xf16, 1>,
        %dst: memref<?x?xf16>
    ) -> () {
        %host_desc = builtin.unrealized_conversion_cast %dst : memref<?x?xf16> to !llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %host_ptr = llvm.extractvalue %host_desc[1] : !llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>

        %dev_desc = builtin.unrealized_conversion_cast %src : memref<?x?xf16, 1> to !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %dev_ptr3 = llvm.extractvalue %dev_desc[1] : !llvm.struct<(!llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.array<2 x i64>, !llvm.array<2 x i64>)>
        %dev_ptr = llvm.addrspacecast %dev_ptr3 : !llvm.ptr<1> to !llvm.ptr

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %m = memref.dim %src, %c0 : memref<?x?xf16, 1>
        %n = memref.dim %src, %c1 : memref<?x?xf16, 1>
        %num_elem = arith.muli %m, %n : index
        %elem_size = arith.constant 2 : index
        %bytes = arith.muli %num_elem, %elem_size : index
        %bytes_i64 = arith.index_cast %bytes : index to i64

        %kind = arith.constant 2 : i32
        %err = llvm.call @cudaMemcpy(%host_ptr, %dev_ptr, %bytes_i64, %kind) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
        return
    }

}
