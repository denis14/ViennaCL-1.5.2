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

/* Computation of eigenvalues of a small symmetric, tridiagonal matrix */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include  <algorithm>

// includes, project

#include "config.hpp"
#include "structs.hpp"
#include "matlab.hpp"

// includes, kernels
#include "bisect_kernel_small.cuh"

// includes, file
//#include "bisect_small.cuh"

namespace viennacl
{
  namespace linalg
  {
    ////////////////////////////////////////////////////////////////////////////////
    //! Determine eigenvalues for matrices smaller than MAX_SMALL_MATRIX
    //! @param TimingIterations  number of iterations for timing
    //! @param  input  handles to input data of kernel
    //! @param  result handles to result of kernel
    //! @param  mat_size  matrix size
    //! @param  lg  lower limit of Gerschgorin interval
    //! @param  ug  upper limit of Gerschgorin interval
    //! @param  precision  desired precision of eigenvalues
    //! @param  iterations  number of iterations for timing
    ////////////////////////////////////////////////////////////////////////////////
    void
    computeEigenvaluesSmallMatrix(InputData &input, ResultDataSmall &result,
                                  const unsigned int mat_size,
                                  const float lg, const float ug,
                                  const float precision)
    {
        

        dim3  blocks(1, 1, 1);
        dim3  threads(MAX_THREADS_BLOCK_SMALL_MATRIX, 1, 1);

        bisectKernel<<< blocks, threads >>>(viennacl::linalg::cuda::detail::cuda_arg<float>(input.vcl_a), 
                                            viennacl::linalg::cuda::detail::cuda_arg<float>(input.vcl_b) + 1, 
                                            mat_size,
                                            viennacl::linalg::cuda::detail::cuda_arg<float>(result.vcl_g_left), 
                                            viennacl::linalg::cuda::detail::cuda_arg<float>(result.vcl_g_right),
                                            viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.vcl_g_left_count),
                                            viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.vcl_g_right_count),
                                            lg, ug, 0, mat_size,
                                            precision
                                           );
        

        checkCudaErrors(cudaDeviceSynchronize());

    //    getLastCudaError("Kernel launch failed");

    }


    ////////////////////////////////////////////////////////////////////////////////
    //! Process the result obtained on the device, that is transfer to host and
    //! perform basic sanity checking
    //! @param  input  handles to input data
    //! @param  result  handles to result data
    //! @param  mat_size   matrix size
    //! @param  filename  output filename
    ////////////////////////////////////////////////////////////////////////////////
    void
    processResultSmallMatrix(const InputData &input, ResultDataSmall &result,
                             const unsigned int mat_size,
                             const char *filename)
    {
        // copy data back to host
        std::vector<float> left(mat_size);
        std::vector<unsigned int> left_count(mat_size);
          
        viennacl::copy(result.vcl_g_left, left);
        viennacl::copy(result.vcl_g_left_count, left_count);
    
        for (unsigned int i = 0; i < mat_size; ++i)
        {
            result.std_eigenvalues[left_count[i]] = left[i];
        }
       // save result in matlab format
       // writeTridiagSymMatlab(filename, input.a, input.b+1, eigenvalues, mat_size);
    }
  }
}
