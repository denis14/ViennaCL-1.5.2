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

//namespace viennacl
//{
  //namespace linalg
  //{

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
    computeEigenvaluesLargeMatrix(InputData &input, ResultDataLarge &result,
                                  const unsigned int mat_size,
                                  const float lg, const float ug,  const float precision)
    {
        dim3  blocks(1, 1, 1);
        dim3  threads(MAX_THREADS_BLOCK, 1, 1);
        std::cout << "Start bisectKernelLarge" << std::endl;
        bisectKernelLarge<<< blocks, threads >>>
          (viennacl::linalg::cuda::detail::cuda_arg<float>(input.vcl_a),
           viennacl::linalg::cuda::detail::cuda_arg<float>(input.vcl_b) + 1,
           mat_size,
           lg, ug, 0, mat_size, precision,
           result.g_num_one, result.g_num_blocks_mult,
           viennacl::linalg::cuda::detail::cuda_arg<float>(result.g_left_one),
           viennacl::linalg::cuda::detail::cuda_arg<float>(result.g_right_one),
           viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_pos_one),
           viennacl::linalg::cuda::detail::cuda_arg<float>(result.g_left_mult),
           viennacl::linalg::cuda::detail::cuda_arg<float>(result.g_right_mult),
           viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_left_count_mult),
           viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_right_count_mult),
           viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_blocks_mult),
           viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_blocks_mult_sum)
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
          (viennacl::linalg::cuda::detail::cuda_arg<float>(input.vcl_a),
           viennacl::linalg::cuda::detail::cuda_arg<float>(input.vcl_b) + 1,
           mat_size, num_one_intervals,
           viennacl::linalg::cuda::detail::cuda_arg<float>(result.g_left_one),
           viennacl::linalg::cuda::detail::cuda_arg<float>(result.g_right_one),
           viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_pos_one),
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

        std::cout << "Start bisectKernelLarge_MultIntervals " << std::endl;
        bisectKernelLarge_MultIntervals<<< grid_mult, threads_mult >>>
          (viennacl::linalg::cuda::detail::cuda_arg<float>(input.vcl_a),
           viennacl::linalg::cuda::detail::cuda_arg<float>(input.vcl_b) + 1,
           mat_size,
           viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_blocks_mult),
           viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_blocks_mult_sum),
           viennacl::linalg::cuda::detail::cuda_arg<float>(result.g_left_mult),
           viennacl::linalg::cuda::detail::cuda_arg<float>(result.g_right_mult),
           viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_left_count_mult),
           viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_right_count_mult),
           viennacl::linalg::cuda::detail::cuda_arg<float>(result.g_lambda_mult),
           viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_pos_mult),
           precision
          );
        viennacl::linalg::cuda::VIENNACL_CUDA_LAST_ERROR_CHECK("bisectKernelLarge_MultIntervals() FAILED.");
        checkCudaErrors(cudaDeviceSynchronize());



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
                                 const char *filename)
    {
        bool bCompareResult = true;
        std::cout << "Matrix size: " << mat_size << std::endl;

        // copy data from intervals that contained more than one eigenvalue after
        // the first processing step
        std::vector<float> lambda_mult(mat_size);
        viennacl::copy(result.g_lambda_mult, lambda_mult);

        std::vector<unsigned int> pos_mult(mat_size);
        viennacl::copy(result.g_pos_mult, pos_mult);

        std::vector<unsigned int> blocks_mult_sum(mat_size);
        viennacl::copy(result.g_blocks_mult_sum, blocks_mult_sum);

        unsigned int num_one_intervals;
        checkCudaErrors(cudaMemcpy(&num_one_intervals, result.g_num_one,
                                   sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));

        unsigned int sum_blocks_mult = mat_size - num_one_intervals;


        // copy data for intervals that contained one eigenvalue after the first
        // processing step
        std::vector<float> left_one(mat_size);
        std::vector<float> right_one(mat_size);
        std::vector<unsigned int> pos_one(mat_size);

        viennacl::copy(result.g_left_one, left_one);
        viennacl::copy(result.g_right_one, right_one);
        viennacl::copy(result.g_pos_one, pos_one);


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

        // store result
        //writeTridiagSymMatlab(filename, input.vcl_a, input.vcl_b, result.std_eigenvalues, mat_size);
        // getLastCudaError( sdkWriteFilef( filename, eigenvals, mat_size, 0.0f));

        return bCompareResult;

    }
  //}
//}
