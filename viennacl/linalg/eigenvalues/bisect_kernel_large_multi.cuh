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

/* Perform second step of bisection algorithm for large matrices for
 * intervals that contained after the first step more than one eigenvalue
 */

#ifndef _BISECT_KERNEL_LARGE_MULTI_H_
#define _BISECT_KERNEL_LARGE_MULTI_H_

// includes, project
#include "config.hpp"
#include "util.hpp"

// additional kernel
#include "bisect_util.cu"

////////////////////////////////////////////////////////////////////////////////
//! Perform second step of bisection algorithm for large matrices for
//! intervals that after the first step contained more than one eigenvalue
//! @param  g_d  diagonal elements of symmetric, tridiagonal matrix
//! @param  g_s  superdiagonal elements of symmetric, tridiagonal matrix
//! @param  n    matrix size
//! @param  blocks_mult  start addresses of blocks of intervals that are
//!                      processed by one block of threads, each of the
//!                      intervals contains more than one eigenvalue
//! @param  blocks_mult_sum  total number of eigenvalues / singleton intervals
//!                          in one block of intervals
//! @param  g_left  left limits of intervals
//! @param  g_right  right limits of intervals
//! @param  g_left_count  number of eigenvalues less than left limits
//! @param  g_right_count  number of eigenvalues less than right limits
//! @param  g_lambda  final eigenvalue
//! @param  g_pos  index of eigenvalue (in ascending order)
//! @param  precision  desired precision of eigenvalues
////////////////////////////////////////////////////////////////////////////////
__global__
void
bisectKernelLarge_MultIntervals(float *g_d, float *g_s, const unsigned int n,
                                unsigned int *blocks_mult,
                                unsigned int *blocks_mult_sum,
                                float *g_left, float *g_right,
                                unsigned int *g_left_count,
                                unsigned int *g_right_count,
                                float *g_lambda, unsigned int *g_pos,
                                float precision
                               )
{
#if 1
  const unsigned int tid = threadIdx.x;

    // left and right limits of interval
    __shared__  float  s_left[2 * MAX_THREADS_BLOCK];
    __shared__  float  s_right[2 * MAX_THREADS_BLOCK];

    // number of eigenvalues smaller than interval limits
    __shared__  unsigned int  s_left_count[2 * MAX_THREADS_BLOCK];
    __shared__  unsigned int  s_right_count[2 * MAX_THREADS_BLOCK];

    // helper array for chunk compaction of second chunk
    __shared__  unsigned int  s_compaction_list[2 * MAX_THREADS_BLOCK + 1];
    // compaction list helper for exclusive scan
    unsigned int *s_compaction_list_exc = s_compaction_list + 1;

    // flag if all threads are converged
    __shared__  unsigned int  all_threads_converged;
    // number of active threads
    __shared__  unsigned int  num_threads_active;
    // number of threads to employ for compaction
    __shared__  unsigned int  num_threads_compaction;
    // flag if second chunk has to be compacted
    __shared__  unsigned int compact_second_chunk;

    // parameters of block of intervals processed by this block of threads
    __shared__  unsigned int  c_block_start;
    __shared__  unsigned int  c_block_end;
    __shared__  unsigned int  c_block_offset_output;

    // midpoint of currently active interval of the thread
    float mid = 0.0f;
    // number of eigenvalues smaller than \a mid
    unsigned int  mid_count = 0;
    // current interval parameter
    float  left;
    float  right;
    unsigned int  left_count;
    unsigned int  right_count;
    // helper for compaction, keep track which threads have a second child
    unsigned int  is_active_second = 0;

    // initialize common start conditions
    if (0 == tid)
    {

        c_block_start = blocks_mult[blockIdx.x];
        c_block_end = blocks_mult[blockIdx.x + 1];
        c_block_offset_output = blocks_mult_sum[blockIdx.x];

        num_threads_active = c_block_end - c_block_start;
        s_compaction_list[0] = 0;
        num_threads_compaction = ceilPow2(num_threads_active);

        all_threads_converged = 1;
        compact_second_chunk = 0;
    }

    __syncthreads();
    
    s_right_count[tid] = 0.0f;                 // selbst hinzugefuegt
    s_left_count[tid] = 0.0f;                 // selbst hinzugefuegt   

    // read data into shared memory
    if (tid < num_threads_active)
    {

        s_left[tid]  = g_left[c_block_start + tid];
        s_right[tid] = g_right[c_block_start + tid];
        s_left_count[tid]  = g_left_count[c_block_start + tid];
        s_right_count[tid] = g_right_count[c_block_start + tid];
        
    }
    printf("1: s_r_c = %u \t s_l_c = %u\n", s_right_count[tid], s_left_count[tid]);       // selbst hinzugefuegt

    __syncthreads();

    // do until all threads converged
    while (true)
    {
        //for (int iter=0; iter < 0; iter++) {
        s_compaction_list[threadIdx.x] = 0;
        s_compaction_list[threadIdx.x + blockDim.x] = 0;
        s_compaction_list[2 * MAX_THREADS_BLOCK] = 0;

        // subdivide interval if currently active and not already converged
        subdivideActiveInterval(tid, s_left, s_right,
                                s_left_count, s_right_count,
                                num_threads_active,
                                left, right, left_count, right_count,
                                mid, all_threads_converged);

        __syncthreads();

        // stop if all eigenvalues have been found
        if (1 == all_threads_converged)
        {
            break;
        }

        // compute number of eigenvalues smaller than mid for active and not
        // converged intervals, use all threads for loading data from gmem and
        // s_left and s_right as scratch space to store the data load from gmem
        // in shared memory
        mid_count = computeNumSmallerEigenvalsLarge(g_d, g_s, n,
                                                    mid, tid, num_threads_active,
                                                    s_left, s_right,
                                                    (left == right));

        __syncthreads();

        if (tid < num_threads_active)
        {

            // store intervals
            if (left != right)
            {

                storeNonEmptyIntervals(tid, num_threads_active,
                                       s_left, s_right, s_left_count, s_right_count,
                                       left, mid, right,
                                       left_count, mid_count, right_count,
                                       precision, compact_second_chunk,
                                       s_compaction_list_exc,
                                       is_active_second);
               // printf("2!!!: s_r_c = %u\n", s_right_count[tid]);                       // selbst hinzugefuegt
            }
            else
            {

                storeIntervalConverged(s_left, s_right, s_left_count, s_right_count,
                                       left, mid, right,
                                       left_count, mid_count, right_count,
                                       s_compaction_list_exc, compact_second_chunk,
                                       num_threads_active,
                                       is_active_second);
               //  printf("3: s_r_c = %u\n", s_right_count[tid]);                       // selbst hinzugefuegt

            }
        }

        __syncthreads();

        // compact second chunk of intervals if any of the threads generated
        // two child intervals
        if (1 == compact_second_chunk)
        {

            createIndicesCompaction(s_compaction_list_exc, num_threads_compaction);

            compactIntervals(s_left, s_right, s_left_count, s_right_count,
                             mid, right, mid_count, right_count,
                             s_compaction_list, num_threads_active,
                             is_active_second);
          //  printf("4: s_r_c = %u\n", s_right_count[tid]);                            // selbst hinzugefuegt
        }

        __syncthreads();

        // update state variables
        if (0 == tid)
        {
            num_threads_active += s_compaction_list[num_threads_active];
            num_threads_compaction = ceilPow2(num_threads_active);

            compact_second_chunk = 0;
            all_threads_converged = 1;
        }

        __syncthreads();

        // clear
        s_compaction_list_exc[threadIdx.x] = 0;
        s_compaction_list_exc[threadIdx.x + blockDim.x] = 0;
        
        if (num_threads_compaction > blockDim.x)              // selbst hinzugefuegt
        {
          break;
        }


        __syncthreads();

    }  // end until all threads converged

    // write data back to global memory
    if (tid < num_threads_active)
    {

        unsigned int addr = c_block_offset_output + tid;
        
       // printf("c_block_offset_output = %u\ts_r_c = %u\n", c_block_offset_output, s_right_count[tid]);        // selbst hinzugefuegt
        g_lambda[addr]  = s_left[tid];
        g_pos[addr]   = s_right_count[tid];
    }
#endif
}

#endif // #ifndef _BISECT_KERNEL_LARGE_MULTI_H_
