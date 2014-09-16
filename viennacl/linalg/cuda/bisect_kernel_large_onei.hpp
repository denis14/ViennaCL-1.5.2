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

/* Determine eigenvalues for large matrices for intervals that contained after
 * the first step one eigenvalue
 */


#ifndef VIENNACL_LINALG_DETAIL_BISECT_KERNEL_LARGE_ONEI_HPP_
#define VIENNACL_LINALG_DETAIL_BISECT_KERNEL_LARGE_ONEI_HPP_

// includes, project
#include "viennacl/linalg/detail/bisect/config.hpp"
#include "viennacl/linalg/detail/bisect/util.hpp"
// additional kernel
#include "viennacl/linalg/cuda/bisect_util.hpp"

namespace viennacl
{
  namespace linalg
  {
    namespace cuda
      {
      ////////////////////////////////////////////////////////////////////////////////
      //! Determine eigenvalues for large matrices for intervals that after
      //! the first step contained one eigenvalue
      //! @param  g_d  diagonal elements of symmetric, tridiagonal matrix
      //! @param  g_s  superdiagonal elements of symmetric, tridiagonal matrix
      //! @param  n    matrix size
      //! @param  num_intervals  total number of intervals containing one eigenvalue
      //!                         after the first step
      //! @param g_left  left interval limits
      //! @param g_right  right interval limits
      //! @param g_pos  index of interval / number of intervals that are smaller than
      //!               right interval limit
      //! @param  precision  desired precision of eigenvalues
      ////////////////////////////////////////////////////////////////////////////////
      template<typename NumericT>
      __global__
      void
      bisectKernelLarge_OneIntervals(const NumericT *g_d, const NumericT *g_s, const unsigned int n,
                                     unsigned int num_intervals,
                                     NumericT *g_left, NumericT *g_right,
                                     unsigned int *g_pos,
                                     NumericT  precision)
      {

          const unsigned int gtid = (blockDim.x * blockIdx.x) + threadIdx.x;

          __shared__  NumericT  s_left_scratch[MAX_THREADS_BLOCK];
          __shared__  NumericT  s_right_scratch[MAX_THREADS_BLOCK];

          // active interval of thread
          // left and right limit of current interval
          NumericT left, right;
          // number of threads smaller than the right limit (also corresponds to the
          // global index of the eigenvalues contained in the active interval)
          unsigned int right_count;
          // flag if current thread converged
          unsigned int converged = 0;
          // midpoint when current interval is subdivided
          NumericT mid = 0.0f;
          // number of eigenvalues less than mid
          unsigned int mid_count = 0;

          // read data from global memory
          if (gtid < num_intervals)
          {
              left = g_left[gtid];
              right = g_right[gtid];
              right_count = g_pos[gtid];
          }


          // flag to determine if all threads converged to eigenvalue
          __shared__  unsigned int  converged_all_threads;

          // initialized shared flag
          if (0 == threadIdx.x)
          {
              converged_all_threads = 0;
          }

          __syncthreads();

          // process until all threads converged to an eigenvalue
          // while( 0 == converged_all_threads) {
          //for (unsigned int i = 0; i < 5; ++i)                        // selbst hinzugefuegt
          while (true)
          {

              converged_all_threads = 1;

              // update midpoint for all active threads
              if ((gtid < num_intervals) && (0 == converged))
              {

                  mid = computeMidpoint(left, right);
              }

              // find number of eigenvalues that are smaller than midpoint
              mid_count = computeNumSmallerEigenvalsLarge(g_d, g_s, n,
                                                          mid, gtid, num_intervals,
                                                          s_left_scratch,
                                                          s_right_scratch,
                                                          converged);

              __syncthreads();

              // for all active threads
              if ((gtid < num_intervals) && (0 == converged))
              {

                  // udpate intervals -- always one child interval survives
                  if (right_count == mid_count)
                  {
                      right = mid;
                  }
                  else
                  {
                      left = mid;
                  }

                  // check for convergence
                  NumericT t0 = right - left;
                  NumericT t1 = max(abs(right), abs(left)) * precision;

                  if (t0 < min(precision, t1))
                  {

                      NumericT lambda = computeMidpoint(left, right);
                      left = lambda;
                      right = lambda;

                      converged = 1;
                  }
                  else
                  {
                      converged_all_threads = 0;
                  }
              }

              __syncthreads();

              if (1 == converged_all_threads)
              {
                  break;
              }

              __syncthreads();
          }

          // write data back to global memory
          __syncthreads();

          if (gtid < num_intervals)
          {
              // intervals converged so left and right interval limit are both identical
              // and identical to the eigenvalue
              g_left[gtid] = left;
          }
      }
    }
  }
}
#endif // #ifndef _BISECT_KERNEL_LARGE_ONEI_H_
