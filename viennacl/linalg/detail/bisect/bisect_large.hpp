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


#ifndef VIENNACL_LINALG_DETAIL_BISECT_LARGE_HPP_
#define VIENNACL_LINALG_DETAIL_BISECT_LARGE_HPP_


/* Computation of eigenvalues of a large symmetric, tridiagonal matrix */

// includes, system
#include <iostream>
#include <iomanip>  
#include <stdlib.h>
#include <stdio.h>

// includes, project
#include "viennacl/linalg/detail/bisect/config.hpp"
#include "viennacl/linalg/detail/bisect/structs.hpp"



#include "viennacl/linalg/detail/bisect/bisect_kernel_calls.hpp"

namespace viennacl
{
  namespace linalg
  {
    namespace detail
    {

      ////////////////////////////////////////////////////////////////////////////////
      //! Run the kernels to compute the eigenvalues for large matrices
      //! @param  input   handles to input data
      //! @param  result  handles to result data
      //! @param  mat_size  matrix size
      //! @param  precision  desired precision of eigenvalues
      //! @param  lg  lower limit of Gerschgorin interval
      //! @param  ug  upper limit of Gerschgorin interval
      ////////////////////////////////////////////////////////////////////////////////
      template<typename NumericT>
      void
      computeEigenvaluesLargeMatrix(InputData<NumericT> &input, ResultDataLarge<NumericT> &result,
                                    const unsigned int mat_size,
                                    const NumericT lg, const NumericT ug,  const NumericT precision)
      {
         // First kernel call
          viennacl::linalg::detail::bisectLarge(input, result, mat_size, lg, ug, precision);

          // compute eigenvalues for intervals that contained only one eigenvalue
          // after the first processing step
          viennacl::linalg::detail::bisectLarge_OneIntervals(input, result, mat_size, precision);

          // process intervals that contained more than one eigenvalue after
          // the first processing step
          viennacl::linalg::detail::bisectLarge_MultIntervals(input, result, mat_size, precision);

      }

      ////////////////////////////////////////////////////////////////////////////////
      //! Process the result, that is obtain result from device and do simple sanity
      //! checking
      //! @param  result  handles to result data
      //! @param  mat_size  matrix size
      ////////////////////////////////////////////////////////////////////////////////
      template<typename NumericT>
      bool
      processResultDataLargeMatrix(ResultDataLarge<NumericT> &result,
                                   const unsigned int mat_size)
      {
          bool bCompareResult = true;
          std::cout << "Matrix size: " << mat_size << std::endl;

          // copy data from intervals that contained more than one eigenvalue after
          // the first processing step
          std::vector<float> lambda_mult(mat_size);
          viennacl::copy(result.g_lambda_mult, lambda_mult);

          std::vector<unsigned int> pos_mult(mat_size);
          viennacl::copy(result.g_pos_mult, pos_mult);

          std::vector<unsigned int> blocks_mult_sum(mat_size);
          viennacl::copy(result.g_blocks_mult_sum, blocks_mult_sum);

          unsigned int num_one_intervals = result.g_num_one;
          unsigned int sum_blocks_mult = mat_size - num_one_intervals;


          // copy data for intervals that contained one eigenvalue after the first
          // processing step
          std::vector<float> left_one(mat_size);
          std::vector<float> right_one(mat_size);
          std::vector<unsigned int> pos_one(mat_size);

          viennacl::copy(result.g_left_one, left_one);
          viennacl::copy(result.g_right_one, right_one);
          viennacl::copy(result.g_pos_one, pos_one);


          // singleton intervals generated in the second step
          for (unsigned int i = 0; i < sum_blocks_mult; ++i)
          {
            if (pos_mult[i] != 0)
              result.std_eigenvalues[pos_mult[i] - 1] = lambda_mult[i];

            else
            {
              printf("pos_mult[%u] = %u\n", i, pos_mult[i]);
              bCompareResult = false;
            }
          }

          // singleton intervals generated in the first step
          unsigned int index = 0;

          for (unsigned int i = 0; i < num_one_intervals; ++i, ++index)
          {
              result.std_eigenvalues[pos_one[i] - 1] = left_one[i];
          }

          return bCompareResult;
      }
    } // namespace detail
  }  // namespace linalg
}  // namespace viennacl
#endif //VIENNACL_LINALG_DETAIL_BISECT_LARGE_HPP_
