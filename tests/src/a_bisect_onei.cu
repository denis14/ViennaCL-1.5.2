/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Computation of eigenvalues of symmetric, tridiagonal matrix using
 * bisection.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

// includes, project
//#include <helper_functions.h>
//#include <helper_cuda.h>

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"

#include "viennacl/linalg/eigenvalues/config.hpp"
#include "viennacl/linalg/eigenvalues/structs.hpp"
#include "viennacl/linalg/eigenvalues/matlab.hpp"
#include "viennacl/linalg/eigenvalues/util.hpp"
#include "viennacl/linalg/eigenvalues/gerschgorin.hpp"
#include "viennacl/linalg/eigenvalues/bisect_large.hpp"

#include "viennacl/linalg/eigenvalues/bisect_large.cuh"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    bool bQAResults = false;

    printf("Starting eigenvalues\n");

    bQAResults = runTest(argc, argv);
    printf("Test %s\n", bQAResults ? "Succeeded!" : "Failed!");

    exit(bQAResults ? EXIT_SUCCESS : EXIT_FAILURE);
}



////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool
runTest(int argc, char **argv)
{
    bool bCompareResult = false;
    // default
    unsigned int mat_size = 520;
    // flag if the matrix size is due to explicit user request
    unsigned int user_defined = 0;
    // desired precision of eigenvalues
    float  precision = 0.00001f;
    unsigned int iters_timing = 1;
    char  *result_file = "eigenvalues.dat";

    {
    // set up input
    InputData input(argv[0], mat_size, user_defined);

    // compute Gerschgorin interval
    float lg = FLT_MAX;
    float ug = -FLT_MAX;
    computeGerschgorin(input.a, input.b + 1, mat_size, lg, ug);
    printf("Gerschgorin interval: %f / %f\n", lg, ug);


    // initialize memory for result
    ResultDataLarge result(mat_size);
    std::cout << "initResultDataLargeMatrix" << std::endl;
    initResultDataLargeMatrix(result, mat_size);


    dim3  blocks(1, 1, 1);
    dim3  threads(MAX_THREADS_BLOCK, 1, 1);

    std::cout << " Start computation of the eigenvalues! " << std::endl;

//anfang bisect.hpp

    std::cout << "Start bisectKernelLarge\n" << std::endl;
    bisectKernelLarge<<< blocks, threads >>>
    (input.g_a, input.g_b, mat_size,
      lg, ug, 0, mat_size, precision,
     result.g_num_one, result.g_num_blocks_mult,
     result.g_left_one, result.g_right_one, result.g_pos_one,
     result.g_left_mult, result.g_right_mult,
     result.g_left_count_mult, result.g_right_count_mult,
     result.g_blocks_mult, result.g_blocks_mult_sum
    );

    viennacl::linalg::cuda::VIENNACL_CUDA_LAST_ERROR_CHECK("Kernel launch failed.");
    checkCudaErrors(cudaDeviceSynchronize());


    // get the number of intervals containing one eigenvalue after the first
    // processing step
    unsigned int num_one_intervals  = 42;
    checkCudaErrors(cudaMemcpy(&num_one_intervals, result.g_num_one,
                               sizeof(unsigned int),
                               cudaMemcpyDeviceToHost));

    dim3 grid_onei;
    grid_onei.x = getNumBlocksLinear(num_one_intervals, MAX_THREADS_BLOCK);
    grid_onei.y = 1, grid_onei.z = 1;
    dim3 threads_onei(MAX_THREADS_BLOCK, 1, 1);
    // use always max number of available threads to better balance load times
    // for matrix data

    // compute eigenvalues for intervals that contained only one eigenvalue
    // after the first processing step

    std::cout << "Start bisectKernelLarge_OneIntervals" << std::endl;
    bisectKernelLarge_OneIntervals<<< grid_onei , threads_onei >>>
    (input.g_a, input.g_b, mat_size, num_one_intervals,
     result.g_left_one, result.g_right_one, result.g_pos_one,
     precision
    );

    viennacl::linalg::cuda::VIENNACL_CUDA_LAST_ERROR_CHECK("bisectKernelLarge_OneIntervals() FAILED.");
    checkCudaErrors(cudaDeviceSynchronize());

// ende von bisect.hpp

    // cleanup
    std::cout << "CleanupResultDataLargeMatrix!" << std::endl;
    cleanupResultDataLargeMatrix(result);

    std::cout << "cleanupInputData" << std::endl;
    input.cleanupInputData();
    }
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    
    std::cout << "cudaDeviceReset" << std::endl;
    cudaDeviceReset();

    return bCompareResult;
}

