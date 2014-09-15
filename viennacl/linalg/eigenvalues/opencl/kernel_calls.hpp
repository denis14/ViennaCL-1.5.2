/* Determine eigenvalues for small symmetric, tridiagonal matrix */

#ifndef _BISECT_OPENCL_KERNEL_CALLS_H_
#define _BISECT_OPENCL_KERNEL_CALLS_H_

// includes, project
#include "viennacl/linalg/eigenvalues/opencl/bisect_kernel_large.hpp"

namespace viennacl
{
  namespace linalg
  {
    namespace opencl
    {


    const std::string BISECT_KERNEL_SMALL = "bisectKernel";
    const std::string BISECT_KERNEL_LARGE = "bisectKernelLarge";
    const std::string BISECT_KERNEL_LARGE_ONE_INTERVALS  = "bisectKernelLarge_OneIntervals";
    const std::string BISECT_KERNEL_LARGE_MULT_INTERVALS = "bisectKernelLarge_MultIntervals";

    void bisect_small_opencl(InputData &input, ResultDataSmall &result,
                       const unsigned int mat_size,
                       const float lg, const float ug,
                       const float precision)
        {
          viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input.vcl_a).context());
          viennacl::linalg::opencl::kernels::bisect_kernel_large<float>::init(ctx);

          viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::bisect_kernel_large<float>::program_name(), BISECT_KERNEL_SMALL);
          kernel.global_work_size(0, 1 * MAX_THREADS_BLOCK_SMALL_MATRIX);
          kernel.local_work_size(0, MAX_THREADS_BLOCK_SMALL_MATRIX);

          viennacl::ocl::enqueue(kernel(
                                        viennacl::traits::opencl_handle(input.vcl_a),
                                        viennacl::traits::opencl_handle(input.vcl_b),  // +1
                                        static_cast<cl_uint>(mat_size),
                                        viennacl::traits::opencl_handle(result.vcl_g_left),
                                        viennacl::traits::opencl_handle(result.vcl_g_right),
                                        viennacl::traits::opencl_handle(result.vcl_g_left_count),
                                        viennacl::traits::opencl_handle(result.vcl_g_right_count),
                                        static_cast<float>(lg),
                                        static_cast<float>(ug),
                                        static_cast<cl_uint>(0),
                                        static_cast<cl_uint>(mat_size),
                                        static_cast<float>(precision)
                                ));

        }

    void bisectLarge_opencl(InputData &input, ResultDataLarge &result,
                       const unsigned int mat_size,
                       const float lg, const float ug,
                       const float precision)
        {
          viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input.vcl_a).context());
          viennacl::linalg::opencl::kernels::bisect_kernel_large<float>::init(ctx);

          viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::bisect_kernel_large<float>::program_name(), BISECT_KERNEL_LARGE);
          kernel.global_work_size(0, 1 * MAX_THREADS_BLOCK);
          kernel.local_work_size(0, MAX_THREADS_BLOCK);

          viennacl::ocl::enqueue(kernel(
                                        viennacl::traits::opencl_handle(input.vcl_a),
                                        viennacl::traits::opencl_handle(input.vcl_b),  // +1
                                        static_cast<cl_uint>(mat_size),
                                        static_cast<float>(lg),
                                        static_cast<float>(ug),
                                        static_cast<cl_uint>(0),
                                        static_cast<cl_uint>(mat_size),
                                        static_cast<float>(precision),
                                        viennacl::traits::opencl_handle(result.g_num_one),
                                        viennacl::traits::opencl_handle(result.g_num_blocks_mult),
                                        viennacl::traits::opencl_handle(result.g_left_one),
                                        viennacl::traits::opencl_handle(result.g_right_one),
                                        viennacl::traits::opencl_handle(result.g_pos_one),
                                        viennacl::traits::opencl_handle(result.g_left_mult),
                                        viennacl::traits::opencl_handle(result.g_right_mult),
                                        viennacl::traits::opencl_handle(result.g_left_count_mult),
                                        viennacl::traits::opencl_handle(result.g_right_count_mult),
                                        viennacl::traits::opencl_handle(result.g_blocks_mult),
                                        viennacl::traits::opencl_handle(result.g_blocks_mult_sum)
                                ));

        }


    void bisectLargeOneIntervals_opencl(InputData &input, ResultDataLarge &result,
                       const unsigned int mat_size,
                       const float precision)
        {
          unsigned int num_one_intervals = result.g_num_one;
          unsigned int num_blocks = getNumBlocksLinear(num_one_intervals, MAX_THREADS_BLOCK);

          viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input.vcl_a).context());
          viennacl::linalg::opencl::kernels::bisect_kernel_large<float>::init(ctx);

          viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::bisect_kernel_large<float>::program_name(), BISECT_KERNEL_LARGE_ONE_INTERVALS);
          kernel.global_work_size(0, num_blocks * MAX_THREADS_BLOCK);
          kernel.local_work_size(0, MAX_THREADS_BLOCK);

          viennacl::ocl::enqueue(kernel(
                                        viennacl::traits::opencl_handle(input.vcl_a),
                                        viennacl::traits::opencl_handle(input.vcl_b),  // will be shifted +1 in kernel
                                        static_cast<cl_uint>(mat_size),
                                        static_cast<cl_uint>(num_one_intervals),
                                        viennacl::traits::opencl_handle(result.g_left_one),
                                        viennacl::traits::opencl_handle(result.g_right_one),
                                        viennacl::traits::opencl_handle(result.g_pos_one),
                                        static_cast<float>(precision)
                                ));

        }


    void bisectLargeMultIntervals_opencl(InputData &input, ResultDataLarge &result,
                       const unsigned int mat_size,
                       const float precision)
        {
          unsigned int  num_blocks_mult = result.g_num_blocks_mult;

          viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input.vcl_a).context());
          viennacl::linalg::opencl::kernels::bisect_kernel_large<float>::init(ctx);

          viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::bisect_kernel_large<float>::program_name(), BISECT_KERNEL_LARGE_MULT_INTERVALS);
          kernel.global_work_size(0, num_blocks_mult * MAX_THREADS_BLOCK);
          kernel.local_work_size(0, MAX_THREADS_BLOCK);

          viennacl::ocl::enqueue(kernel(
                                        viennacl::traits::opencl_handle(input.vcl_a),
                                        viennacl::traits::opencl_handle(input.vcl_b),  // will be shifted +1 in kernel
                                        static_cast<cl_uint>(mat_size),
                                        viennacl::traits::opencl_handle(result.g_blocks_mult),
                                        viennacl::traits::opencl_handle(result.g_blocks_mult_sum),
                                        viennacl::traits::opencl_handle(result.g_left_mult),
                                        viennacl::traits::opencl_handle(result.g_right_mult),
                                        viennacl::traits::opencl_handle(result.g_left_count_mult),
                                        viennacl::traits::opencl_handle(result.g_right_count_mult),
                                        viennacl::traits::opencl_handle(result.g_lambda_mult),
                                        viennacl::traits::opencl_handle(result.g_pos_mult),
                                        static_cast<float>(precision)
                                ));
        }

    }
  }
}

#endif
