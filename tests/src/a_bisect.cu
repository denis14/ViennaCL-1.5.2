/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/*
*
*   Test file for tql-algorithm
*
*/

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

// include necessary system headers
#include <iostream>

#ifndef NDEBUG
  #define NDEBUG
#endif

#define VIENNACL_WITH_UBLAS



#include "viennacl/linalg/qr-method.hpp"

#define EPS 10.0e-5



namespace ublas = boost::numeric::ublas;
typedef float     ScalarType;


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
    unsigned int mat_size = 120;
    // flag if the matrix size is due to explicit user request
    unsigned int user_defined = 0;
    // desired precision of eigenvalues
    float  precision = 0.00001f;
    char  *result_file = "eigenvalues.dat";

    {
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
    std::cout << "initResultDataLargeMatrix" << std::endl;
    initResultDataLargeMatrix(result, mat_size);
/*
    // run the kernel
    computeEigenvaluesLargeMatrix(input, result, mat_size,
                                  precision, lg, ug,
                                  iters_timing);

   

*/

    dim3  blocks(1, 1, 1);
    dim3  threads(MAX_THREADS_BLOCK, 1, 1);

    std::cout << " Start computation of the eigenvalues! " << std::endl;

// anfang von bisect.hpp

    std::cout << "Start bisectKernelLarge" << std::endl;
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

    // process intervals that contained more than one eigenvalue after
    // the first processing step

    // get the number of blocks of intervals that contain, in total when
    // each interval contains only one eigenvalue, not more than
    // MAX_THREADS_BLOCK threads
    unsigned int  num_blocks_mult = 0;
    checkCudaErrors(cudaMemcpy(&num_blocks_mult, result.g_num_blocks_mult,
                               sizeof(unsigned int),
                               cudaMemcpyDeviceToHost));

    // setup the execution environment
    dim3  grid_mult(num_blocks_mult, 1, 1);
    dim3  threads_mult(MAX_THREADS_BLOCK, 1, 1);


    std::cout << "Start bisectKernelLarge_MultIntervals\t num_blocks_mult = " << num_blocks_mult << std::endl;
    bisectKernelLarge_MultIntervals<<< grid_mult, threads_mult >>>
    (input.g_a, input.g_b, mat_size,
     result.g_blocks_mult, result.g_blocks_mult_sum,
     result.g_left_mult, result.g_right_mult,
     result.g_left_count_mult, result.g_right_count_mult,
     result.g_lambda_mult, result.g_pos_mult,
     precision
    );
    viennacl::linalg::cuda::VIENNACL_CUDA_LAST_ERROR_CHECK("bisectKernelLarge_MultIntervals() FAILED.");
    checkCudaErrors(cudaDeviceSynchronize());

// ende bisect.hpp

     // get the result from the device and do some sanity checks
    // save the result if user specified matrix size
    bCompareResult = processResultDataLargeMatrix(input, result, mat_size, result_file,
                                                  user_defined, argv[0]);
                                                  
                                                  
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



/**
 * Test the tql2 algorithm for symmetric tridiagonal matrices.
 */
/*
template <typename MatrixLayout>
void test_qr_method_sym()
{
  std::size_t sz = 10;

  viennacl::matrix<ScalarType, MatrixLayout> Q = viennacl::identity_matrix<ScalarType>(sz);
  ublas::vector<ScalarType> d(sz), e(sz), d_ref(sz), e_ref(sz); //d is major diagonal, e is minor diagonal

  std::cout << "Testing matrix of size " << sz << "-by-" << sz << std::endl << std::endl;

  for(unsigned int i = 0; i < sz; ++i)
  {
    //std_a[i] = i % 11 + 4;
    //std_b_raw[i] = i % 9 + 2;
    d[i] = ((float)(i % 9)) - 4.5f;
    e[i] = ((float)(i % 5)) - 4.5f;
  }
  e[0] = 0.0f;
  d_ref = d;
  e_ref = e;

//--------------------------------------------------------
  viennacl::linalg::tql2(Q, d, e);

 // if(!test_eigen_val_vec<MatrixLayout>(Q, d, d_ref, e_ref))
   //  exit(EXIT_FAILURE);

  for( unsigned int i = 0; i < sz; ++i)
    std::cout << "Eigenvalue " << i << "= " << d[i] << std::endl;
}
*/
/*
int main()
{

  std::cout << std::endl << "Testing tql2 algorithm for symmetric tridiagonal row-major matrices..." << std::endl;
  test_qr_method_sym<viennacl::row_major>();
/*
  std::cout << std::endl << "Testing QL algorithm for symmetric tridiagonal column-major matrices..." << std::endl;
  test_qr_method_sym<viennacl::column_major>();
*/
  //std::cout << std::endl <<"--------TEST SUCCESSFULLY COMPLETED----------" << std::endl;
//}
