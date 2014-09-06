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

#ifndef NDEBUG
  #define NDEBUG
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
//#include <math.h>
//#include <float.h>
//#include <assert.h>

// includes, project
//#include <helper_functions.h>
//#include <helper_cuda.h>

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
//#include "viennacl/compressed_matrix.hpp"

#include "viennacl/linalg/eigenvalues/config.hpp"
#include "viennacl/linalg/eigenvalues/structs.hpp"
#include "viennacl/linalg/eigenvalues/matlab.hpp"
#include "viennacl/linalg/eigenvalues/util.hpp"
#include "viennacl/linalg/eigenvalues/gerschgorin.hpp"
#include "viennacl/linalg/eigenvalues/bisect_large.hpp"

#include "viennacl/linalg/eigenvalues/bisect_large.cuh"
#include "viennacl/linalg/eigenvalues/bisect_small.cuh"
#include "viennacl/linalg/eigenvalues/bisect_small.cu"

#include "viennacl/linalg/qr-method.hpp"

#define EPS 10.0e-5

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



void
initInputData(std::vector<float> &diagonal, std::vector<float> &superdiagonal, const unsigned int mat_size)
{
 // initialize diagonal and superdiagonal entries with random values
  srand(278217421);

  // srand( clock());
  /*
  for (unsigned int i = 0; i < mat_size; ++i)
  {
      diagonal[i] = (float)(2.0 * (((double)rand()
                                   / (double) RAND_MAX) - 0.5));
      superdiagonal[i] = (float)(2.0 * (((double)rand()
                                   / (double) RAND_MAX) - 0.5));
  }*/
  
  for(unsigned int i = 0; i < mat_size; ++i)
    {
     diagonal[i] = ((float)(i % 9)) - 4.5f;
     superdiagonal[i] = ((float)(i % 5)) - 4.5f;
    }
 
}
template <typename NumericT>
bool bisect(const std::vector<NumericT> & diagonal, const std::vector<NumericT> & superdiagonal, std::vector<NumericT> & eigenvalues, const unsigned int mat_size)
{
    bool bCompareResult = false;
    // flag if the matrix size is due to explicit user request
    // desired precision of eigenvalues
    float  precision = 0.00001f;
    char  *result_file = "eigenvalues.dat";
    
    // set up input
    InputData input(diagonal, superdiagonal, mat_size);
    // compute Gerschgorin interval
    float lg = FLT_MAX;
    float ug = -FLT_MAX;
    computeGerschgorin(input.a, input.b + 1, mat_size, lg, ug);
    //computeGerschgorin(input.std_a, input.std_b_raw, mat_size, lg, ug);
    printf("Gerschgorin interval: %f / %f\n", lg, ug);
    
    if (mat_size <= MAX_SMALL_MATRIX)
    {

      // initialize memory for result
      ResultDataSmall result(mat_size);
      initResultSmallMatrix(result, mat_size);

      // run the kernel
      computeEigenvaluesSmallMatrix(input, result, mat_size, lg, ug,
                                    precision, 1);

      // get the result from the device and do some sanity checks,
      // save the result
      processResultSmallMatrix(input, result, mat_size, result_file);
      std::copy(result.std_eigenvalues.begin(), result.std_eigenvalues.end(), eigenvalues.begin());
      
      // clean up
      cleanupResultSmallMatrix(result);
      bCompareResult = true;
    }

    else
    {
      // initialize memory for result
      ResultDataLarge result(mat_size);
      initResultDataLargeMatrix(result, mat_size);

      // run the kernel
      computeEigenvaluesLargeMatrix(input, result, mat_size,
                                    precision, lg, ug,
                                    1);

     
       // get the result from the device and do some sanity checks
      // save the result if user specified matrix size
      bCompareResult = processResultDataLargeMatrix(input, result, mat_size, result_file);
                                                    
      std::copy(result.std_eigenvalues.begin(), result.std_eigenvalues.end(), eigenvalues.begin());                                         
      // cleanup
      std::cout << "CleanupResultDataLargeMatrix!" << std::endl;
      cleanupResultDataLargeMatrix(result);
    } //Large end
    
    std::cout << "cleanupInputData" << std::endl;
    input.cleanupInputData();
    return bCompareResult;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool
runTest(int argc, char **argv)
{
    bool bCompareResult = false;
    // default
    unsigned int mat_size = 2203;
    
    std::vector<float> diagonal(mat_size);
    std::vector<float> superdiagonal(mat_size);
    std::vector<float> eigenvalues_bisect(mat_size);
    
    // Fill the vectors diagonal and superdiagonal
    initInputData(diagonal, superdiagonal, mat_size);
    
    //Start the bisection algorithm
    std::cout << "Start the bisection algorithm" << std::endl;
    bCompareResult = bisect(diagonal, superdiagonal, eigenvalues_bisect, mat_size);
    
    if (bCompareResult == false)
     return false;
  /*  
    for (unsigned int i = 0; i < mat_size; ++i)
    {
      std::cout << "Eigenvalue " << i << ": " << std::setprecision(10) << eigenvalues_bisect[i] << std::endl;
    }
    */
    // The results of the bisection algorithm will be checked with the tql2 algorithm
    // Initialize Data for tql2 algorithm
    viennacl::matrix<float, viennacl::row_major> Q = viennacl::identity_matrix<float>(mat_size);
    std::vector<float> diagonal_tql(mat_size);
    std::vector<float> superdiagonal_tql(mat_size);
    diagonal_tql = diagonal;
    superdiagonal_tql = superdiagonal;
    
    std::cout << "Start the tql2 algorithm..." << std::endl; 
    viennacl::linalg::tql2(Q, diagonal_tql, superdiagonal_tql);  
    
    std::cout << "Start sorting..." << std::endl;
    std::sort(diagonal_tql.begin(), diagonal_tql.end());
    
    std::cout << "Start comparison..." << std::endl;
    for(uint i = 0; i < mat_size; i++)
    {
       if(std::abs(eigenvalues_bisect[i] - diagonal_tql[i]) > EPS)
       { 
	       std::cout << std::setprecision(8) << eigenvalues_bisect[i] << "  != " << diagonal_tql[i] << "\n";
	       return EXIT_FAILURE;
       }  	
    }
    std::cout << "mat_size = " << mat_size << std::endl;
    for (unsigned int i = 0; i < mat_size; ++i)
    {
      std::cout << "Eigenvalue " << i << ": \tbisect: " << std::setprecision(8) << eigenvalues_bisect[i] << "\ttql2: " << diagonal_tql[i] << std::endl;
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

