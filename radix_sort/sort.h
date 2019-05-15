#ifndef SORT_H__
#define SORT_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "scan.h"
#include <cmath>

void radix_sort(float* const d_out,
    unsigned int* const d_idx_out,
    float* const d_in,
    unsigned int d_in_len);

#endif