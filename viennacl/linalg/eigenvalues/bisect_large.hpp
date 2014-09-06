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

/* Computation of eigenvalues of a large symmetric, tridiagonal matrix */

// includes, system
#include <iostream>
#include <iomanip>  
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, project
//#include "helper_functions.h"
//#include "helper_cuda.h"
#include "config.hpp"
#include "structs.hpp"
#include "util.hpp"
#include "matlab.hpp"

#include "bisect_large.cuh"

// includes, kernels
#include "bisect_kernel_large.cuh"
#include "bisect_kernel_large_onei.cuh"
#include "bisect_kernel_large_multi.cuh"



#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"

#include "viennacl/linalg/cuda/common.hpp"

#include "viennacl/linalg/cuda/vector_operations.hpp"
#include "viennacl/linalg/cuda/matrix_operations_row.hpp"
#include "viennacl/linalg/cuda/matrix_operations_col.hpp"
#include "viennacl/linalg/cuda/matrix_operations_prod.hpp"
#include "viennacl/linalg/cuda/matrix_operations_prod.hpp"

////////////////////////////////////////////////////////////////////////////////
//! Initialize variables and memory for result
//! @param  result handles to memory
//! @param  matrix_size  size of the matrix
////////////////////////////////////////////////////////////////////////////////
void
initResultDataLargeMatrix(ResultDataLarge &result, const unsigned int mat_size)
{

    // helper variables to initialize memory
    unsigned int zero = 0;
    unsigned int mat_size_f = sizeof(float) * mat_size;
    unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;

    float *tempf = (float *) malloc(mat_size_f);
    unsigned int *tempui = (unsigned int *) malloc(mat_size_ui);

    for (unsigned int i = 0; i < mat_size; ++i)
    {
        tempf[i] = 0.0f;
        tempui[i] = 0;
    }

    // number of intervals containing only one eigenvalue after the first step
    checkCudaErrors(cudaMalloc((void **) &result.g_num_one,
                               sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(result.g_num_one, &zero, sizeof(unsigned int),
                               cudaMemcpyHostToDevice));

    // number of (thread) blocks of intervals with multiple eigenvalues after
    // the first iteration
    checkCudaErrors(cudaMalloc((void **) &result.g_num_blocks_mult,
                               sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(result.g_num_blocks_mult, &zero,
                               sizeof(unsigned int),
                               cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMalloc((void **) &result.g_left_one, mat_size_f));
    checkCudaErrors(cudaMalloc((void **) &result.g_right_one, mat_size_f));
    checkCudaErrors(cudaMalloc((void **) &result.g_pos_one, mat_size_ui));

    checkCudaErrors(cudaMalloc((void **) &result.g_left_mult, mat_size_f));
    checkCudaErrors(cudaMalloc((void **) &result.g_right_mult, mat_size_f));
    checkCudaErrors(cudaMalloc((void **) &result.g_left_count_mult,
                               mat_size_ui));
    checkCudaErrors(cudaMalloc((void **) &result.g_right_count_mult,
                               mat_size_ui));

    checkCudaErrors(cudaMemcpy(result.g_left_one, tempf, mat_size_f,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(result.g_right_one, tempf, mat_size_f,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(result.g_pos_one, tempui, mat_size_ui,
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(result.g_left_mult, tempf, mat_size_f,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(result.g_right_mult, tempf, mat_size_f,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(result.g_left_count_mult, tempui, mat_size_ui,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(result.g_right_count_mult, tempui, mat_size_ui,
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &result.g_blocks_mult, mat_size_ui));
    checkCudaErrors(cudaMemcpy(result.g_blocks_mult, tempui, mat_size_ui,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &result.g_blocks_mult_sum, mat_size_ui));
    checkCudaErrors(cudaMemcpy(result.g_blocks_mult_sum, tempui, mat_size_ui,
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &result.g_lambda_mult, mat_size_f));
    checkCudaErrors(cudaMemcpy(result.g_lambda_mult, tempf, mat_size_f,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &result.g_pos_mult, mat_size_ui));
    checkCudaErrors(cudaMemcpy(result.g_pos_mult, tempf, mat_size_ui,
                               cudaMemcpyHostToDevice));

}

////////////////////////////////////////////////////////////////////////////////
//! Cleanup result memory
//! @param result  handles to memory
////////////////////////////////////////////////////////////////////////////////
void
cleanupResultDataLargeMatrix(ResultDataLarge &result)
{

    checkCudaErrors(cudaFree(result.g_num_one));
    checkCudaErrors(cudaFree(result.g_num_blocks_mult));
    checkCudaErrors(cudaFree(result.g_left_one));
    checkCudaErrors(cudaFree(result.g_right_one));
    checkCudaErrors(cudaFree(result.g_pos_one));
    checkCudaErrors(cudaFree(result.g_left_mult));
    checkCudaErrors(cudaFree(result.g_right_mult));
    checkCudaErrors(cudaFree(result.g_left_count_mult));
    checkCudaErrors(cudaFree(result.g_right_count_mult));
    checkCudaErrors(cudaFree(result.g_blocks_mult));
    checkCudaErrors(cudaFree(result.g_blocks_mult_sum));
    checkCudaErrors(cudaFree(result.g_lambda_mult));
    checkCudaErrors(cudaFree(result.g_pos_mult));
}

////////////////////////////////////////////////////////////////////////////////
//! Run the kernels to compute the eigenvalues for large matrices
//! @param  input   handles to input data
//! @param  result  handles to result data
//! @param  mat_size  matrix size
//! @param  precision  desired precision of eigenvalues
//! @param  lg  lower limit of Gerschgorin interval
//! @param  ug  upper limit of Gerschgorin interval
//! @param  iterations  number of iterations (for timing)
////////////////////////////////////////////////////////////////////////////////
void
computeEigenvaluesLargeMatrix(const InputData &input, const ResultDataLarge &result,
                              const unsigned int mat_size, const float precision,
                              const float lg, const float ug,
                              const unsigned int iterations)
{
    dim3  blocks(1, 1, 1);
    dim3  threads(MAX_THREADS_BLOCK, 1, 1);

    std::cout << " Start computation of the eigenvalues! " << std::endl;


    for( unsigned int i = 0; i < 10; ++i)
      std::cout << "a " << i << "= " << input.a[i] << std::endl;

    for( unsigned int i = 0; i < 10; ++i)
      std::cout << "b " << i << "= " << input.b[i] << std::endl;


    // do for multiple iterations to improve timing accuracy
    for (unsigned int iter = 0; iter < iterations; ++iter)
    {

        std::cout << "Start bisectKernelLarge\t iter = " << iter << std::endl;
        bisectKernelLarge<<< blocks, threads >>>
        (input.g_a, input.g_b, mat_size,
          lg, ug, 0, mat_size, precision,
         result.g_num_one, result.g_num_blocks_mult,
         result.g_left_one, result.g_right_one, result.g_pos_one,
         result.g_left_mult, result.g_right_mult,
         result.g_left_count_mult, result.g_right_count_mult,
         result.g_blocks_mult, result.g_blocks_mult_sum
        );

       // viennacl::linalg::cuda::VIENNACL_CUDA_LAST_ERROR_CHECK("Kernel launch failed.");
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
        //threads_onei.x = MAX_THREADS_BLOCK;

        // compute eigenvalues for intervals that contained only one eigenvalue
        // after the first processing step

         //grid_onei.x = 1;
        // std::cout << "grid_onei.x, y, z: " << grid_onei.x << "  " << grid_onei.y << "  " << grid_onei.z << std::endl;
        //std::cout << "thread_onei.x, y, z: " << threads_onei.x << "  " << threads_onei.y << "  " << threads_onei.z << std::endl;
        
        std::cout << "Start bisectKernelLarge_OneIntervals" << std::endl;
        bisectKernelLarge_OneIntervals<<< grid_onei , threads_onei >>>
        (input.g_a, input.g_b, mat_size, num_one_intervals,
         result.g_left_one, result.g_right_one, result.g_pos_one,
         precision
        );

       // viennacl::linalg::cuda::VIENNACL_CUDA_LAST_ERROR_CHECK("bisectKernelLarge_OneIntervals() FAILED.");
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

        //grid_mult.x = 1;
      //  std::cout << "grid_mult.x, y, z: " << grid_mult.x << "  " << grid_mult.y << "  " << grid_mult.z << std::endl;
    //   std::cout << "thread_mult.x, y, z: " << threads_mult.x << "  " << threads_mult.y << "  " << threads_mult.z << std::endl;
        
        std::cout << "Start bisectKernelLarge_MultIntervals: num_block_mult = " << num_blocks_mult << std::endl;
        bisectKernelLarge_MultIntervals<<< grid_mult, threads_mult >>>
        (input.g_a, input.g_b, mat_size,
         result.g_blocks_mult, result.g_blocks_mult_sum,
         result.g_left_mult, result.g_right_mult,
         result.g_left_count_mult, result.g_right_count_mult,
         result.g_lambda_mult, result.g_pos_mult,
         precision
        );
      //  viennacl::linalg::cuda::VIENNACL_CUDA_LAST_ERROR_CHECK("bisectKernelLarge_MultIntervals() FAILED.");
        checkCudaErrors(cudaDeviceSynchronize());

    }

}

////////////////////////////////////////////////////////////////////////////////
//! Process the result, that is obtain result from device and do simple sanity
//! checking
//! @param  input   handles to input data
//! @param  result  handles to result data
//! @param  mat_size  matrix size
//! @param  filename  output filename
////////////////////////////////////////////////////////////////////////////////
bool
processResultDataLargeMatrix(const InputData &input, ResultDataLarge &result,
                             const unsigned int mat_size,
                             const char *filename,
                             const unsigned int user_defined, char *exec_path)
{
    bool bCompareResult = true;
    std::cout << "Matrix size: " << mat_size << std::endl;
    const unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;
    const unsigned int mat_size_f  = sizeof(float) * mat_size;

    // copy data from intervals that contained more than one eigenvalue after
    // the first processing step
    float *lambda_mult = (float *) malloc(sizeof(float) * mat_size);
    checkCudaErrors(cudaMemcpy(lambda_mult, result.g_lambda_mult,
                               sizeof(float) * mat_size,
                               cudaMemcpyDeviceToHost));
    unsigned int *pos_mult =
        (unsigned int *) malloc(sizeof(unsigned int) * mat_size);
    checkCudaErrors(cudaMemcpy(pos_mult, result.g_pos_mult,
                               sizeof(unsigned int) * mat_size,
                               cudaMemcpyDeviceToHost));

    unsigned int *blocks_mult_sum =
        (unsigned int *) malloc(sizeof(unsigned int) * mat_size);
    checkCudaErrors(cudaMemcpy(blocks_mult_sum, result.g_blocks_mult_sum,
                               sizeof(unsigned int) * mat_size,
                               cudaMemcpyDeviceToHost));

    unsigned int num_one_intervals;
    checkCudaErrors(cudaMemcpy(&num_one_intervals, result.g_num_one,
                               sizeof(unsigned int),
                               cudaMemcpyDeviceToHost));

    unsigned int sum_blocks_mult = mat_size - num_one_intervals;


    // copy data for intervals that contained one eigenvalue after the first
    // processing step
    float *left_one = (float *) malloc(mat_size_f);
    float *right_one = (float *) malloc(mat_size_f);
    unsigned int *pos_one = (unsigned int *) malloc(mat_size_ui);
    checkCudaErrors(cudaMemcpy(left_one, result.g_left_one, mat_size_f,
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(right_one, result.g_right_one, mat_size_f,
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pos_one, result.g_pos_one, mat_size_ui,
                               cudaMemcpyDeviceToHost));

    // extract eigenvalues
   // viennacl::vector<float> eigenvals(mat_size);
    // extract eigenvalues
    float *eigenvalues = (float *) malloc(mat_size_f);


    // singleton intervals generated in the second step
    for (unsigned int i = 0; i < sum_blocks_mult; ++i)
    {
      
      
      if (pos_mult[i] != 0)
        result.std_eigenvalues[pos_mult[i] - 1] = lambda_mult[i];
      
      else
      {
        printf("pos_mult[%u] = %u\n", i, pos_mult[i]);
        bCompareResult = false;
      } 
      }

    // singleton intervals generated in the first step
    unsigned int index = 0;

    for (unsigned int i = 0; i < num_one_intervals; ++i, ++index)
    {
        result.std_eigenvalues[pos_one[i] - 1] = left_one[i];
    }

    for( unsigned int i = 0; i < mat_size; ++i)
      std::cout << "Eigenvalue " << i << "= " << std::setprecision(10) << result.std_eigenvalues[i] << std::endl;

    if (0 == user_defined)
    {
        // store result
        //writeTridiagSymMatlab(filename, input.vcl_a, input.vcl_b, result.std_eigenvalues, mat_size);
        // getLastCudaError( sdkWriteFilef( filename, eigenvals, mat_size, 0.0f));

        printf("skipping self-check!\n");
    }

    freePtr(eigenvalues);
    freePtr(lambda_mult);
    freePtr(pos_mult);
    freePtr(blocks_mult_sum);
    freePtr(left_one);
    freePtr(right_one);
    freePtr(pos_one);

    return bCompareResult;
    
}

