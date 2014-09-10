/* Determine eigenvalues for small symmetric, tridiagonal matrix */

#ifndef _BISECT_CUDA_KERNEL_CALLS_H_
#define _BISECT_CUDA_KERNEL_CALLS_H_

// includes, project
#include "viennacl/linalg/eigenvalues/cuda/bisect_kernel_small.cuh"

namespace viennacl
{
  namespace linalg
  {
    namespace cuda
    {

      void bisect_small_cuda(InputData &input, ResultDataSmall &result,
                         const unsigned int mat_size,
                         const float lg, const float ug,
                         const float precision)
      {


        dim3  blocks(1, 1, 1);
        dim3  threads(MAX_THREADS_BLOCK_SMALL_MATRIX, 1, 1);

        bisectKernelSmall<<< blocks, threads >>>(viennacl::linalg::cuda::detail::cuda_arg<float>(input.vcl_a),
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

        viennacl::linalg::cuda::VIENNACL_CUDA_LAST_ERROR_CHECK("Kernel launch failed");


      }



    }
  }
}






#endif
