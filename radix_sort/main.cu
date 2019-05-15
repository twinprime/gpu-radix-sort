#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <math.h>

#include "sort.h"
#include "utils.h"

void cpu_sort(float* h_out, float* h_in, size_t len)
{
    for (unsigned int i = 0; i < len; ++i)
    {
        h_out[i] = h_in[i];
    }

    std::sort(h_out, h_out + len);
}

int main()
{
    srand(1);

    unsigned int N = 10000;

    for (int i = 0; i < 5; i++) {
        std::cout << "N = " << N << " ----------" << std::endl;

        // float h_in[N] = { 0.08680988, -0.44326124, -0.15096481,  0.68955225, -0.99056226, -0.75686175,
        //     0.34149817,  0.6517055,  -0.72658682,  0.15018666,  0.78264391, -0.58159578,
        //    -0.62934357, -0.78324622, -0.56060499,  0.95724756,  0.6233663 };
        // float h_in[N] = {
        //     8.68098810e-02,  -4.43261236e-01,  -1.50964811e-01,   6.89552248e-01,
        //     -9.90562260e-01,  -7.56861746e-01,   3.41498166e-01,   6.51705503e-01,
        //     -7.26586819e-01,   1.50186658e-01,   7.82643914e-01,  -5.81595778e-01,
        //     -6.29343569e-01,  -7.83246219e-01,  -5.60604990e-01,   9.57247555e-01,
        //      6.23366296e-01,  -6.56117976e-01,   6.32449508e-01,  -4.51852500e-01,
        //     -1.36591628e-01,   8.80059659e-01,   6.35298729e-01,  -3.27776104e-01,
        //     -6.49179101e-01,  -2.54335910e-01,  -9.88622963e-01,  -4.95147288e-01,
        //      5.91325045e-01,  -9.69490051e-01,   1.97686747e-01,   2.07609072e-01,
        //     -7.89704621e-01,  -2.36113116e-01,  -9.27047908e-01,   7.80823112e-01,
        //      9.61841702e-01,  -8.80116045e-01,   7.81091869e-01,   1.53803006e-01,
        //      4.84959364e-01,   2.60367870e-01,   1.63684383e-01,  -9.59121764e-01,
        //     -5.79946816e-01,   8.93697590e-02,   5.38230360e-01,  -4.98609543e-01,
        //     -4.28208619e-01,   7.04790175e-01,   9.50012982e-01,   7.69706607e-01,
        //     -2.80984312e-01,   1.97717890e-01,  -2.90408790e-01,  -3.19619566e-01,
        //     -6.43838048e-01,  -5.24611592e-01,  -9.10275459e-01,   1.08628590e-02,
        //     -2.47495085e-01,   1.85610801e-01,   2.59883761e-01,  -7.14799345e-01,
        //      8.67682576e-01,   8.92759740e-01,   2.04593316e-01,  -2.24467441e-01,
        //     -2.73624003e-01,  -5.91309428e-01,  -4.46469873e-01,  -5.06928265e-01,
        //     -6.52783990e-01,   9.33219373e-01,   9.14025187e-01,   1.95947364e-01,
        //      4.62601513e-01,  -3.19229543e-01,  -8.15888822e-01,  -7.30039626e-02,
        //      1.73977856e-02,  -8.23079646e-01,   5.60704470e-02,   9.84316051e-01,
        //     -2.09928140e-01,  -3.28807116e-01,   6.10901058e-01,   5.08697987e-01,
        //     -3.73867124e-01,   2.68073380e-01,   8.08091536e-02,  -4.06412512e-01,
        //     -7.78424203e-01,  -3.74719411e-01,  -8.60417411e-02,   3.17880154e-01,
        //     -4.91484970e-01,   2.82202512e-01,  -5.99752784e-01,   3.15249622e-01,
        //      5.56578457e-01,   5.59196770e-01,   2.20656306e-01,  -3.81999314e-01,
        //      3.95469815e-01,   7.19236612e-01,   2.50647515e-01,   9.64815676e-01,
        //      9.53000247e-01,  -6.66611731e-01,  -9.53643739e-01,  -6.78510904e-01,
        //      8.46993625e-01,   9.07099724e-01,  -5.78043163e-01,  -2.78949499e-01,
        //      9.87505242e-02,  -4.56338316e-01,  -7.87967592e-02,   3.92323136e-01,
        //      7.11793371e-04,   4.32141989e-01,   5.19118719e-02,  -9.97201979e-01,
        //     -2.10599422e-01,  -1.56660601e-02,  -1.94239333e-01,  -2.91403413e-01,
        //      1.22863892e-03
        // };
        float* h_in = new float[N];
        float* h_out = new float[N];
        unsigned int* h_idx_out = new unsigned int[N];

        for (unsigned int i = 0; i < N; i++) {
            h_in[i] = ((float) rand() / RAND_MAX - 0.5f) * 2;
            // std::cout << i << ": " << h_in[i] << std::endl;
        }

        std::clock_t start;
        start = std::clock();

        float* d_in;
        float* d_out;
        unsigned int* d_idx_out = NULL;
        checkCudaErrors(cudaMalloc(&d_in, sizeof(float) * N));
        checkCudaErrors(cudaMalloc(&d_out, sizeof(float) * N));
        checkCudaErrors(cudaMalloc(&d_idx_out, sizeof(unsigned int) * N));
        checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(float) * N, cudaMemcpyHostToDevice));

        radix_sort(d_out, d_idx_out, d_in, N);
        checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_idx_out, d_idx_out, sizeof(unsigned int) * N, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_out));
        checkCudaErrors(cudaFree(d_idx_out));
        checkCudaErrors(cudaFree(d_in));

        // for (unsigned int i = 0; i < N; i++) {
        //     std::cout << h_idx_out[i] << ": " << h_out[i] << std::endl;
        // }

        double gpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
        std::cout << "GPU time: " << gpu_duration << " s" << std::endl;

        float* h_out_cpu = new float[N];
        start = std::clock();
        cpu_sort(h_out_cpu, h_in, N);
        double cpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
        std::cout << "CPU time: " << cpu_duration << " s" << std::endl;

        bool match = true;
        for (unsigned int i = 0; i < N; ++i)
        {
            if (h_out_cpu[i] != h_out[i])
            {
                match = false;
                break;
            }
        }
        std::cout << "Match: " << match << std::endl;

        N *= 10;
    }
}
