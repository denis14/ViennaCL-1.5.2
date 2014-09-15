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

#ifndef VIENNACL_LINALG_DETAIL_BISECT_SMALL_HPP_
#define VIENNACL_LINALG_DETAIL_BISECT_SMALL_HPP_

/* Computation of eigenvalues of a small symmetric, tridiagonal matrix */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, project

#include "viennacl/linalg/detail/bisect/structs.hpp"

// includes, kernels
#include "viennacl/linalg/detail/bisect/bisect_kernel_calls.hpp"

namespace viennacl
{
  namespace linalg
  {
    namespace detail
    {
      ////////////////////////////////////////////////////////////////////////////////
      //! Determine eigenvalues for matrices smaller than MAX_SMALL_MATRIX
      //! @param  input  handles to input data of kernel
      //! @param  result handles to result of kernel
      //! @param  mat_size  matrix size
      //! @param  lg  lower limit of Gerschgorin interval
      //! @param  ug  upper limit of Gerschgorin interval
      //! @param  precision  desired precision of eigenvalues
      ////////////////////////////////////////////////////////////////////////////////
      template<typename NumericT>
      void
      computeEigenvaluesSmallMatrix(const InputData<NumericT> &input, ResultDataSmall<NumericT> &result,
                                    const unsigned int mat_size,
                                    const NumericT lg, const NumericT ug,
                                    const NumericT precision)
      {
        viennacl::linalg::detail::bisectSmall( input, result, mat_size, lg, ug, precision);
      }


      ////////////////////////////////////////////////////////////////////////////////
      //! Process the result obtained on the device, that is transfer to host and
      //! perform basic sanity checking
      //! @param  result  handles to result data
      //! @param  mat_size   matrix size
      ////////////////////////////////////////////////////////////////////////////////
      template<typename NumericT>
      void
      processResultSmallMatrix(ResultDataSmall<NumericT> &result,
                               const unsigned int mat_size)
      {
        // copy data back to host
        std::vector<NumericT> left(mat_size);
        std::vector<unsigned int> left_count(mat_size);

        viennacl::copy(result.vcl_g_left, left);
        viennacl::copy(result.vcl_g_left_count, left_count);

        for (unsigned int i = 0; i < mat_size; ++i)
        {
            result.std_eigenvalues[left_count[i]] = left[i];
        }
      }
    }  // namespace detail
  }  // namespace linalg
} // namespace viennacl
#endif
