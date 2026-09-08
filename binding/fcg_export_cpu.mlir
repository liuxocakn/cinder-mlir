module {

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
}
