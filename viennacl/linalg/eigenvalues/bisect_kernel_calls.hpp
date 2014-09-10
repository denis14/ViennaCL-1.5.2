#ifndef VIENNACL_LINALG_BISECT_KERNEL_CALLS_HPP_
#define VIENNACL_LINALG_BISECT_KERNEL_CALLS_HPP_

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

/** @file viennacl/linalg/matrix_operations.hpp
    @brief Implementations of dense matrix related operations including matrix-vector products.
*/

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
//#include "viennacl/linalg/host_based/matrix_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
 //  #include "viennacl/linalg/eigenvalues/opencl/kernel_calls.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/eigenvalues/cuda/kernel_calls.hpp"
#endif

namespace viennacl
{
  namespace linalg
  {

   void bisect_small(InputData &input, ResultDataSmall &result,
                      const unsigned int mat_size,
                      const float lg, const float ug,
                      const float precision)
    {
      switch (viennacl::traits::handle(input.vcl_a).get_active_handle_id())
      {
        //    viennacl::linalg::host_based::bisect_small_cuda(input, result,
          //                                       mat_size,
            //                                     lg,ug,
              //                                   precision);
         // break;
#ifdef VIENNACL_WITH_OPENCL
      //    viennacl::linalg::opencl::bisect_small_cuda(input, result,
        //                                       mat_size,
          //                                     lg,ug,
            //                                   precision);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::bisect_small_cuda(input, result,
                                               mat_size,
                                               lg,ug,
                                               precision);

//          std::cout << "Hallo" << std::endl;
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

/*

    template <typename NumericT, typename F>
    void house_update_QL(matrix_base<NumericT, F> & Q,
                         vector_base<NumericT>    & D,
                         vcl_size_t A_size1)
    {
      switch (viennacl::traits::handle(Q).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::house_update_QL(Q, D, A_size1);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::house_update_QL(Q, D, A_size1);
          break;
#endif

#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::house_update_QL(Q, D, A_size1);
          break;
#endif

        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }
*/
  } //namespace linalg

} //namespace viennacl


#endif