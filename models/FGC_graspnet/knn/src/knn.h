#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{
    // TODO: controlla dimensioni dei tensori

    long batch = ref.size(0);
    long dim = ref.size(1);
    long ref_nb = ref.size(2);
    long query_nb = query.size(2);
    long k = idx.size(1);

    float* ref_dev = ref.data_ptr<float>();
    float* query_dev = query.data_ptr<float>();
    long* idx_dev = idx.data_ptr<long>();

    if (ref.is_cuda()) {
#ifdef WITH_CUDA
        // Alloca memoria CUDA per distanze
        float* dist_dev = nullptr;
        cudaError_t err = cudaMalloc((void**)&dist_dev, ref_nb * query_nb * sizeof(float));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            return -1;
        }

        for (int b = 0; b < batch; b++) {
            knn_device(
                ref_dev + b * dim * ref_nb,
                ref_nb,
                query_dev + b * dim * query_nb,
                query_nb,
                dim,
                k,
                dist_dev,
                idx_dev + b * k * query_nb,
                c10::cuda::getCurrentCUDAStream());
        }

        cudaFree(dist_dev);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error after knn_device: %s\n", cudaGetErrorString(err));
            return -1;
        }

        return 1;
#else
        AT_ERROR("Not compiled with CUDA support");
#endif
    }

    // CPU fallback
    float* dist_cpu = (float*)malloc(ref_nb * query_nb * sizeof(float));
    long* ind_buf = (long*)malloc(ref_nb * sizeof(long));

    for (int b = 0; b < batch; b++) {
        knn_cpu(
            ref_dev + b * dim * ref_nb,
            ref_nb,
            query_dev + b * dim * query_nb,
            query_nb,
            dim,
            k,
            dist_cpu,
            idx_dev + b * k * query_nb,
            ind_buf);
    }

    free(dist_cpu);
    free(ind_buf);

    return 1;
}
