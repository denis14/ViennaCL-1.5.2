/* Determine eigenvalues for small symmetric, tridiagonal matrix */

#ifndef _BISECT_OPENCL_KERNEL_CALLS_H_
#define _BISECT_OPENCL_KERNEL_CALLS_H_

// includes, project
#include "viennacl/linalg/eigenvalues/bisect_kernel_small.cuh"

namespace viennacl
{
  namespace linalg
  {
    namespace opencl
    {


/*

    const std::string SVD_GIVENS_NEXT_KERNEL = "givens_next";
    const std::string SVD_COPY_COL_KERNEL = "copy_col";
    const std::string SVD_COPY_ROW_KERNEL = "copy_row";
    const std::string SVD_INCLUSIVE_SCAN_KERNEL_1 = "inclusive_scan_1";
    const std::string SVD_EXCLUSIVE_SCAN_KERNEL_1 = "exclusive_scan_1";
    const std::string SVD_SCAN_KERNEL_2 = "scan_kernel_2";
    const std::string SVD_SCAN_KERNEL_3 = "scan_kernel_3";
    const std::string SVD_SCAN_KERNEL_4 = "scan_kernel_4";

*/

    void bisect_small_opencl(InputData &input, ResultDataSmall &result,
                       const unsigned int mat_size,
                       const float lg, const float ug,
                       const float precision)
        {
       /*   viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input.vcl_a).context());
          viennacl::linalg::opencl::kernels::svd<NumericT, F>::init(ctx);
          viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, F>::program_name(), SVD_GIVENS_NEXT_KERNEL);
          kernel.global_work_size(0, 1);
          kernel.local_work_size(0, MAX_THREADS_BLOCK_SMALL_MATRIX);

          viennacl::ocl::enqueue(kernel(
                                        viennacl::traits::opencl_handle(input.vcl_a),
                                        viennacl::traits::opencl_handle(input.vcl_b) + 1,
                                        static_cast<cl_uint>(mat_size),
                                        viennacl::traits::opencl_handle(result.vcl_g_left),
                                        viennacl::traits::opencl_handle(result.vcl_g_right),
                                        viennacl::traits::opencl_handle(result.vcl_g_left_count),
                                        viennacl::traits::opencl_handle(result.vcl_g_right_count),
                                        static_cast<cl_uint>(lg),
                                        static_cast<cl_uint>(ug),
                                        static_cast<cl_uint>(0),
                                        static_cast<cl_uint>(mat_size),
                                        static_cast<float>(precision)
                                ));*/
        }
/*
        template <typename NumericT, typename F>
        void copy_vec(matrix_base<NumericT, F>& A,
                      vector_base<NumericT> & V,
                      vcl_size_t row_start,
                      vcl_size_t col_start,
                      bool copy_col
        )
        {


          std::string kernel_name = copy_col ? SVD_COPY_COL_KERNEL : SVD_COPY_ROW_KERNEL;
          viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
          viennacl::linalg::opencl::kernels::svd<NumericT, F>::init(ctx);
          viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, F>::program_name(), kernel_name);

          viennacl::ocl::enqueue(kernel(
                                        A,
                                        V,
                                        static_cast<cl_uint>(row_start),
                                        static_cast<cl_uint>(col_start),
                                        copy_col ? cl_uint(viennacl::traits::size1(A))
                                                 : cl_uint(viennacl::traits::size2(A)),
                                        static_cast<cl_uint>(A.internal_size2())
                                ));

        }

#define SECTION_SIZE 256
        template<typename NumericT>
        void inclusive_scan(vector_base<NumericT>& vec1,
                            vector_base<NumericT>& vec2)
        {
          viennacl::vector<NumericT> S( std::ceil(vec1.size() / static_cast<float>(SECTION_SIZE)) ), S_ref( std::ceil(vec1.size() / static_cast<float>(SECTION_SIZE)) );

          viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
          viennacl::linalg::opencl::kernels::svd<NumericT, float>::init(ctx);
          viennacl::ocl::kernel& kernel1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, float>::program_name(), SVD_INCLUSIVE_SCAN_KERNEL_1);
          viennacl::ocl::kernel& kernel2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, float>::program_name(), SVD_SCAN_KERNEL_2);
          viennacl::ocl::kernel& kernel3 = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, float>::program_name(), SVD_SCAN_KERNEL_3);
          viennacl::ocl::kernel& kernel4 = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, float>::program_name(), SVD_SCAN_KERNEL_4);

          kernel1.global_work_size(0, SECTION_SIZE * S.size());
          kernel1.local_work_size(0, SECTION_SIZE);
          viennacl::ocl::enqueue(kernel1(
                                            viennacl::traits::opencl_handle(vec1),
                                            static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                            static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                            static_cast<unsigned int>(viennacl::traits::size(vec1)),

                                            viennacl::traits::opencl_handle(vec2),
                                            static_cast<unsigned int>(viennacl::traits::start(vec2)),
                                            static_cast<unsigned int>(viennacl::traits::stride(vec2)),

                                            viennacl::traits::opencl_handle(S),
                                            static_cast<unsigned int>(viennacl::traits::start(S)),
                                            static_cast<unsigned int>(viennacl::traits::stride(S))));


          kernel2.global_work_size(0, viennacl::tools::align_to_multiple<cl_uint>(cl_uint(viennacl::traits::size(S)), 256));
          kernel2.local_work_size(0, SECTION_SIZE);
          viennacl::ocl::enqueue(kernel2(
                                             viennacl::traits::opencl_handle(S_ref),
                                             static_cast<unsigned int>(viennacl::traits::start(S_ref)),
                                             static_cast<unsigned int>(viennacl::traits::stride(S_ref)),

                                             viennacl::traits::opencl_handle(S),
                                             static_cast<unsigned int>(viennacl::traits::start(S)),
                                             static_cast<unsigned int>(viennacl::traits::stride(S)),
                                             static_cast<unsigned int>(viennacl::traits::size(S))
                                     ));

          kernel3.global_work_size(0,  viennacl::tools::align_to_multiple<cl_uint>(cl_uint(viennacl::traits::size(S)), 256));
          kernel3.local_work_size(0, SECTION_SIZE);
          viennacl::ocl::enqueue(kernel3(
                                             viennacl::traits::opencl_handle(S_ref),
                                             static_cast<unsigned int>(viennacl::traits::start(S_ref)),
                                             static_cast<unsigned int>(viennacl::traits::stride(S_ref)),

                                             viennacl::traits::opencl_handle(S),
                                             static_cast<unsigned int>(viennacl::traits::start(S)),
                                             static_cast<unsigned int>(viennacl::traits::stride(S))
                                     ));


          kernel4.global_work_size(0, SECTION_SIZE * S.size());
          kernel4.local_work_size(0, SECTION_SIZE);
          viennacl::ocl::enqueue(kernel4(
                                            viennacl::traits::opencl_handle(S),
                                            static_cast<unsigned int>(viennacl::traits::start(S)),
                                            static_cast<unsigned int>(viennacl::traits::stride(S)),

                                            viennacl::traits::opencl_handle(vec2),
                                            static_cast<unsigned int>(viennacl::traits::start(vec2)),
                                            static_cast<unsigned int>(viennacl::traits::stride(vec2)),
                                            static_cast<unsigned int>(viennacl::traits::size(vec2))
                                     ));


    }

*/

    } // namespace opencl
  } //namespace linalg
} //namespace viennacl

#endif
