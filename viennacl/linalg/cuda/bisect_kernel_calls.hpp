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


/* Determine eigenvalues for small symmetric, tridiagonal matrix */

#ifndef _BISECT_CUDA_KERNEL_CALLS_H_
#define _BISECT_CUDA_KERNEL_CALLS_H_

// includes, kernels
#include "viennacl/linalg/cuda/bisect_kernel_small.hpp"
#include "viennacl/linalg/cuda/bisect_kernel_large.hpp"
#include "viennacl/linalg/cuda/bisect_kernel_large_onei.hpp"
#include "viennacl/linalg/cuda/bisect_kernel_large_multi.hpp"


namespace viennacl
{
namespace linalg
{
namespace cuda
{
  template<typename NumericT>
  void bisectSmall(const viennacl::linalg::detail::InputData<NumericT> &input, viennacl::linalg::detail::ResultDataSmall<NumericT> &result,
                         const unsigned int mat_size,
                         const NumericT lg, const NumericT ug,
                         const NumericT precision)
  {


    dim3  blocks(1, 1, 1);
    dim3  threads(MAX_THREADS_BLOCK_SMALL_MATRIX, 1, 1);

    bisectKernelSmall<<< blocks, threads >>>(
      viennacl::linalg::cuda::detail::cuda_arg<NumericT>(input.g_a),
      viennacl::linalg::cuda::detail::cuda_arg<NumericT>(input.g_b) + 1,
      mat_size,
      viennacl::linalg::cuda::detail::cuda_arg<NumericT>(result.vcl_g_left),
      viennacl::linalg::cuda::detail::cuda_arg<NumericT>(result.vcl_g_right),
      viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.vcl_g_left_count),
      viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.vcl_g_right_count),
      lg, ug, 0, mat_size,
      precision
      );
    viennacl::linalg::cuda::VIENNACL_CUDA_LAST_ERROR_CHECK("Kernel launch failed");
  }


  template<typename NumericT>
  void bisectLarge(const viennacl::linalg::detail::InputData<NumericT> &input, viennacl::linalg::detail::ResultDataLarge<NumericT> &result,
                     const unsigned int mat_size,
                     const NumericT lg, const NumericT ug,
                     const NumericT precision)
   {

    dim3  blocks(1, 1, 1);
    dim3  threads(MAX_THREADS_BLOCK, 1, 1);
    bisectKernelLarge<<< blocks, threads >>>
      (viennacl::linalg::cuda::detail::cuda_arg<NumericT>(input.g_a),
       viennacl::linalg::cuda::detail::cuda_arg<NumericT>(input.g_b) + 1,
       mat_size,
       lg, ug, 0, mat_size, precision,
       viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_num_one),
       viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_num_blocks_mult),
       viennacl::linalg::cuda::detail::cuda_arg<NumericT>(result.g_left_one),
       viennacl::linalg::cuda::detail::cuda_arg<NumericT>(result.g_right_one),
       viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_pos_one),
       viennacl::linalg::cuda::detail::cuda_arg<NumericT>(result.g_left_mult),
       viennacl::linalg::cuda::detail::cuda_arg<NumericT>(result.g_right_mult),
       viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_left_count_mult),
       viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_right_count_mult),
       viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_blocks_mult),
       viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_blocks_mult_sum)
       );
    viennacl::linalg::cuda::VIENNACL_CUDA_LAST_ERROR_CHECK("Kernel launch failed.");
  }


  // compute eigenvalues for intervals that contained only one eigenvalue
  // after the first processing step
  template<typename NumericT>
  void bisectLarge_OneIntervals(const viennacl::linalg::detail::InputData<NumericT> &input, viennacl::linalg::detail::ResultDataLarge<NumericT> &result,
                     const unsigned int mat_size,
                     const NumericT precision)
   {

    unsigned int num_one_intervals = result.g_num_one;
    unsigned int num_blocks = viennacl::linalg::detail::getNumBlocksLinear(num_one_intervals, MAX_THREADS_BLOCK);
    dim3 grid_onei;
    grid_onei.x = num_blocks;
    grid_onei.y = 1, grid_onei.z = 1;
    dim3 threads_onei(MAX_THREADS_BLOCK, 1, 1);


    bisectKernelLarge_OneIntervals<<< grid_onei , threads_onei >>>
      (viennacl::linalg::cuda::detail::cuda_arg<NumericT>(input.g_a),
       viennacl::linalg::cuda::detail::cuda_arg<NumericT>(input.g_b) + 1,
       mat_size, num_one_intervals,
       viennacl::linalg::cuda::detail::cuda_arg<NumericT>(result.g_left_one),
       viennacl::linalg::cuda::detail::cuda_arg<NumericT>(result.g_right_one),
       viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_pos_one),
       precision
       );
    viennacl::linalg::cuda::VIENNACL_CUDA_LAST_ERROR_CHECK("bisectKernelLarge_OneIntervals() FAILED.");
  }


  // process intervals that contained more than one eigenvalue after
  // the first processing step
  template<typename NumericT>
  void bisectLarge_MultIntervals(const viennacl::linalg::detail::InputData<NumericT> &input, viennacl::linalg::detail::ResultDataLarge<NumericT> &result,
                     const unsigned int mat_size,
                     const NumericT precision)
   {
      // get the number of blocks of intervals that contain, in total when
      // each interval contains only one eigenvalue, not more than
      // MAX_THREADS_BLOCK threads
      unsigned int  num_blocks_mult = result.g_num_blocks_mult;

      // setup the execution environment
      dim3  grid_mult(num_blocks_mult, 1, 1);
      dim3  threads_mult(MAX_THREADS_BLOCK, 1, 1);

      bisectKernelLarge_MultIntervals<<< grid_mult, threads_mult >>>
        (viennacl::linalg::cuda::detail::cuda_arg<NumericT>(input.g_a),
         viennacl::linalg::cuda::detail::cuda_arg<NumericT>(input.g_b) + 1,
         mat_size,
         viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_blocks_mult),
         viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_blocks_mult_sum),
         viennacl::linalg::cuda::detail::cuda_arg<NumericT>(result.g_left_mult),
         viennacl::linalg::cuda::detail::cuda_arg<NumericT>(result.g_right_mult),
         viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_left_count_mult),
         viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_right_count_mult),
         viennacl::linalg::cuda::detail::cuda_arg<NumericT>(result.g_lambda_mult),
         viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_pos_mult),
         precision
        );
      viennacl::linalg::cuda::VIENNACL_CUDA_LAST_ERROR_CHECK("bisectKernelLarge_MultIntervals() FAILED.");
  }
}
}
}

#endif