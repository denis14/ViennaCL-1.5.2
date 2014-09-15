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


#ifdef VIENNACL_WITH_OPENCL
   #include "viennacl/linalg/opencl/bisect_kernel_calls.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/cuda/bisect_kernel_calls.hpp"
#endif

namespace viennacl
{
  namespace linalg
  {

   void bisect_small(const InputData &input, ResultDataSmall &result,
                      const unsigned int mat_size,
                      const float lg, const float ug,
                      const float precision)
    {
      switch (viennacl::traits::handle(input.vcl_a).get_active_handle_id())
      {
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::bisect_small_opencl(input, result,
                                               mat_size,
                                               lg,ug,
                                               precision);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::bisect_small_cuda(input, result,
                                               mat_size,
                                               lg,ug,
                                               precision);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }





   void bisectLarge(const InputData &input, ResultDataLarge &result,
                      const unsigned int mat_size,
                      const float lg, const float ug,
                      const float precision)
    {
      switch (viennacl::traits::handle(input.vcl_a).get_active_handle_id())
      {
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::bisectLarge_opencl(input, result,
                                               mat_size,
                                               lg,ug,
                                               precision);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::bisectLargeCuda(input, result,
                                               mat_size,
                                               lg,ug,
                                               precision);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }






   void bisectLarge_OneIntervals(const InputData &input, ResultDataLarge &result,
                      const unsigned int mat_size,
                      const float precision)
    {
      switch (viennacl::traits::handle(input.vcl_a).get_active_handle_id())
      {
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::bisectLargeOneIntervals_opencl(input, result,
                                               mat_size,
                                               precision);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::bisectLarge_OneIntervalsCuda(input, result,
                                               mat_size,
                                               precision);

          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }





   void bisectLarge_MultIntervals(const InputData &input, ResultDataLarge &result,
                      const unsigned int mat_size,
                      const float precision)
    {
      switch (viennacl::traits::handle(input.vcl_a).get_active_handle_id())
      {
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
        viennacl::linalg::opencl::bisectLargeMultIntervals_opencl(input, result,
                                             mat_size,
                                             precision);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::bisectLarge_MultIntervalsCuda(input, result,
                                               mat_size,
                                               precision);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

  } //namespace linalg

} //namespace viennacl


#endif
