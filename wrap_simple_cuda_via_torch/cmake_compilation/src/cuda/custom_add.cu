__global__ void add_kernel(float* c, const float* a, const float* b, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

void launch_add(float* c, const float* a, const float* b, int n) {
    dim3 grid((n + 1023) / 1024);
    dim3 block(1024);
    add_kernel<<<grid, block>>>(c, a, b, n);
}