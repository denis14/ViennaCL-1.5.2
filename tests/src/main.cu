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
    unsigned int iters_timing = 100;
    char  *result_file = "eigenvalues.dat";

    // set up input
    InputData input(argv[0], mat_size, user_defined);

    // compute Gerschgorin interval
    float lg = FLT_MAX;
    float ug = -FLT_MAX;
    computeGerschgorin(input.a, input.b + 1, mat_size, lg, ug);
    //computeGerschgorin(input.std_a, input.std_b_raw, mat_size, lg, ug);
    printf("Gerschgorin interval: %f / %f\n", lg, ug);


    // initialize memory for result
    ResultDataLarge result(mat_size);
    std::cout << "now initResultDataLargeMatrix" << std::endl;
    initResultDataLargeMatrix(result, mat_size);
/*
    // run the kernel
    computeEigenvaluesLargeMatrix(input, result, mat_size,
                                  precision, lg, ug,
                                  iters_timing);

    // get the result from the device and do some sanity checks
    // save the result if user specified matrix size
    bCompareResult = processResultDataLargeMatrix(input, result, mat_size, result_file,
                                                  user_defined, argv[0]);
*/
    // cleanup
    std::cout << "CleanupResultDataLargeMatrix!" << std::endl;
    cleanupResultDataLargeMatrix(result);

    std::cout << "cleanupInputData" << std::endl;
    input.cleanupInputData();

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    
    std::cout << "cudaDeviceReset" << std::endl;
    cudaDeviceReset();

    return bCompareResult;
}

