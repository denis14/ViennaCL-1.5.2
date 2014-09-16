/* Determine eigenvalues for small symmetric, tridiagonal matrix */

#ifndef VIENNACL_LINALG_DETAIL_BISECT_KERNEL_CALLS_HPP_
#define VIENNACL_LINALG_DETAIL_BISECT_KERNEL_CALLS_HPP_

// includes, project
#include "viennacl/linalg/opencl/kernels/bisect.hpp"

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

    template<typename NumericT>
    void bisectSmall(const viennacl::linalg::detail::InputData<NumericT> &input,
                             viennacl::linalg::detail::ResultDataSmall<NumericT> &result,
                             const unsigned int mat_size,
                             const NumericT lg, const NumericT ug,
                             const NumericT precision)
        {
          viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input.g_a).context());
          viennacl::linalg::opencl::kernels::bisect_kernel_large<NumericT>::init(ctx);

          viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::bisect_kernel_large<NumericT>::program_name(), BISECT_KERNEL_SMALL);
          kernel.global_work_size(0, 1 * MAX_THREADS_BLOCK_SMALL_MATRIX);
          kernel.local_work_size(0, MAX_THREADS_BLOCK_SMALL_MATRIX);

          viennacl::ocl::enqueue(kernel(viennacl::traits::opencl_handle(input.g_a),
                                        viennacl::traits::opencl_handle(input.g_b),
                                        static_cast<cl_uint>(mat_size),
                                        viennacl::traits::opencl_handle(result.vcl_g_left),
                                        viennacl::traits::opencl_handle(result.vcl_g_right),
                                        viennacl::traits::opencl_handle(result.vcl_g_left_count),
                                        viennacl::traits::opencl_handle(result.vcl_g_right_count),
                                        static_cast<NumericT>(lg),
                                        static_cast<NumericT>(ug),
                                        static_cast<cl_uint>(0),
                                        static_cast<cl_uint>(mat_size),
                                        static_cast<NumericT>(precision)
                                ));

        }

    template<typename NumericT>
    void bisectLarge(const viennacl::linalg::detail::InputData<NumericT> &input,
                     viennacl::linalg::detail::ResultDataLarge<NumericT> &result,
                     const unsigned int mat_size,
                     const NumericT lg, const NumericT ug,
                     const NumericT precision)
        {
          viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input.g_a).context());
          viennacl::linalg::opencl::kernels::bisect_kernel_large<NumericT>::init(ctx);

          viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::bisect_kernel_large<NumericT>::program_name(), BISECT_KERNEL_LARGE);
          kernel.global_work_size(0, 1 * MAX_THREADS_BLOCK);
          kernel.local_work_size(0, MAX_THREADS_BLOCK);

          viennacl::ocl::enqueue(kernel(viennacl::traits::opencl_handle(input.g_a),
                                        viennacl::traits::opencl_handle(input.g_b),
                                        static_cast<cl_uint>(mat_size),
                                        static_cast<NumericT>(lg),
                                        static_cast<NumericT>(ug),
                                        static_cast<cl_uint>(0),
                                        static_cast<cl_uint>(mat_size),
                                        static_cast<NumericT>(precision),
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

    template<typename NumericT>
    void bisectLargeOneIntervals(const viennacl::linalg::detail::InputData<NumericT> &input,
                                 viennacl::linalg::detail::ResultDataLarge<NumericT> &result,
                                 const unsigned int mat_size,
                                 const NumericT precision)
        {
          unsigned int num_one_intervals = result.g_num_one;
          unsigned int num_blocks = viennacl::linalg::detail::getNumBlocksLinear(num_one_intervals, MAX_THREADS_BLOCK);

          viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input.g_a).context());
          viennacl::linalg::opencl::kernels::bisect_kernel_large<NumericT>::init(ctx);

          viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::bisect_kernel_large<NumericT>::program_name(), BISECT_KERNEL_LARGE_ONE_INTERVALS);
          kernel.global_work_size(0, num_blocks * MAX_THREADS_BLOCK);
          kernel.local_work_size(0, MAX_THREADS_BLOCK);

          viennacl::ocl::enqueue(kernel(viennacl::traits::opencl_handle(input.g_a),
                                        viennacl::traits::opencl_handle(input.g_b),
                                        static_cast<cl_uint>(mat_size),
                                        static_cast<cl_uint>(num_one_intervals),
                                        viennacl::traits::opencl_handle(result.g_left_one),
                                        viennacl::traits::opencl_handle(result.g_right_one),
                                        viennacl::traits::opencl_handle(result.g_pos_one),
                                        static_cast<NumericT>(precision)
                                ));
        }


    template<typename NumericT>
    void bisectLargeMultIntervals(const viennacl::linalg::detail::InputData<NumericT> &input,
                                  viennacl::linalg::detail::ResultDataLarge<NumericT> &result,
                                  const unsigned int mat_size,
                                  const NumericT precision)
        {
          unsigned int  num_blocks_mult = result.g_num_blocks_mult;

          viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input.g_a).context());
          viennacl::linalg::opencl::kernels::bisect_kernel_large<NumericT>::init(ctx);

          viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::bisect_kernel_large<NumericT>::program_name(), BISECT_KERNEL_LARGE_MULT_INTERVALS);
          kernel.global_work_size(0, num_blocks_mult * MAX_THREADS_BLOCK);
          kernel.local_work_size(0, MAX_THREADS_BLOCK);

          viennacl::ocl::enqueue(kernel(viennacl::traits::opencl_handle(input.g_a),
                                        viennacl::traits::opencl_handle(input.g_b),
                                        static_cast<cl_uint>(mat_size),
                                        viennacl::traits::opencl_handle(result.g_blocks_mult),
                                        viennacl::traits::opencl_handle(result.g_blocks_mult_sum),
                                        viennacl::traits::opencl_handle(result.g_left_mult),
                                        viennacl::traits::opencl_handle(result.g_right_mult),
                                        viennacl::traits::opencl_handle(result.g_left_count_mult),
                                        viennacl::traits::opencl_handle(result.g_right_count_mult),
                                        viennacl::traits::opencl_handle(result.g_lambda_mult),
                                        viennacl::traits::opencl_handle(result.g_pos_mult),
                                        static_cast<NumericT>(precision)
                                ));
        }
    } // namespace opencl
  } // namespace linalg
} // namespace viennacl

#endif
