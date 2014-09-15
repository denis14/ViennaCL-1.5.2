/* Determine eigenvalues for small symmetric, tridiagonal matrix */

#ifndef _BISECT_CUDA_KERNEL_CALLS_H_
#define _BISECT_CUDA_KERNEL_CALLS_H_

// includes, kernels
#include "viennacl/linalg/cuda/bisect_kernel_small.cuh"
#include "viennacl/linalg/cuda/bisect_kernel_large.cuh"
#include "viennacl/linalg/cuda/bisect_kernel_large_onei.cuh"
#include "viennacl/linalg/cuda/bisect_kernel_large_multi.cuh"


namespace viennacl
{
  namespace linalg
  {
    namespace cuda
    {
      void bisect_small_cuda(const InputData &input, ResultDataSmall &result,
                             const unsigned int mat_size,
                             const float lg, const float ug,
                             const float precision)
      {


        dim3  blocks(1, 1, 1);
        dim3  threads(MAX_THREADS_BLOCK_SMALL_MATRIX, 1, 1);

        bisectKernelSmall<<< blocks, threads >>>(
          viennacl::linalg::cuda::detail::cuda_arg<float>(input.vcl_a),
          viennacl::linalg::cuda::detail::cuda_arg<float>(input.vcl_b) + 1,
          mat_size,
          viennacl::linalg::cuda::detail::cuda_arg<float>(result.vcl_g_left),
          viennacl::linalg::cuda::detail::cuda_arg<float>(result.vcl_g_right),
          viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.vcl_g_left_count),
          viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.vcl_g_right_count),
          lg, ug, 0, mat_size,
          precision
          );
        viennacl::linalg::cuda::VIENNACL_CUDA_LAST_ERROR_CHECK("Kernel launch failed");
      }


      void bisectLargeCuda(const InputData &input, ResultDataLarge &result,
                         const unsigned int mat_size,
                         const float lg, const float ug,
                         const float precision)
       {

        dim3  blocks(1, 1, 1);
        dim3  threads(MAX_THREADS_BLOCK, 1, 1);
        std::cout << "Start bisectKernelLarge" << std::endl;
        bisectKernelLarge<<< blocks, threads >>>
          (viennacl::linalg::cuda::detail::cuda_arg<float>(input.vcl_a),
           viennacl::linalg::cuda::detail::cuda_arg<float>(input.vcl_b) + 1,
           mat_size,
           lg, ug, 0, mat_size, precision,
           viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_num_one),
           viennacl::linalg::cuda::detail::cuda_arg<unsigned int>(result.g_num_blocks_mult),
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
      }


      // compute eigenvalues for intervals that contained only one eigenvalue
      // after the first processing step
      void bisectLarge_OneIntervalsCuda(const InputData &input, ResultDataLarge &result,
                         const unsigned int mat_size,
                         const float precision)
       {

        unsigned int num_one_intervals = result.g_num_one;
        unsigned int num_blocks = getNumBlocksLinear(num_one_intervals, MAX_THREADS_BLOCK);
        dim3 grid_onei;
        grid_onei.x = num_blocks;
        grid_onei.y = 1, grid_onei.z = 1;
        dim3 threads_onei(MAX_THREADS_BLOCK, 1, 1);

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
      }


      // process intervals that contained more than one eigenvalue after
      // the first processing step
      void bisectLarge_MultIntervalsCuda(const InputData &input, ResultDataLarge &result,
                         const unsigned int mat_size,
                         const float precision)
       {
          // get the number of blocks of intervals that contain, in total when
          // each interval contains only one eigenvalue, not more than
          // MAX_THREADS_BLOCK threads
          unsigned int  num_blocks_mult = result.g_num_blocks_mult;

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
      }
    }
  }
}






#endif
