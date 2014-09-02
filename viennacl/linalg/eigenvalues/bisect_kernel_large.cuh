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

/* Determine eigenvalues for large symmetric, tridiagonal matrix. First
  step of the computation. */

#ifndef _BISECT_KERNEL_LARGE_H_
#define _BISECT_KERNEL_LARGE_H_

// includes, project
#include "config.hpp"
#include "util.hpp"

// additional kernel
#include "bisect_util.cu"

// declaration, forward

////////////////////////////////////////////////////////////////////////////////
//! Write data to global memory
////////////////////////////////////////////////////////////////////////////////
__device__
void writeToGmem(const unsigned int tid, const unsigned int tid_2,
                 const unsigned int num_threads_active,
                 const unsigned int num_blocks_mult,
                 float *g_left_one, float *g_right_one,
                 unsigned int *g_pos_one,
                 float *g_left_mult, float *g_right_mult,
                 unsigned int *g_left_count_mult,
                 unsigned int *g_right_count_mult,
                 float *s_left, float *s_right,
                 unsigned short *s_left_count, unsigned short *s_right_count,
                 unsigned int *g_blocks_mult,
                 unsigned int *g_blocks_mult_sum,
                 unsigned short *s_compaction_list,
                 unsigned short *s_cl_helper,
                 unsigned int offset_mult_lambda
                );

////////////////////////////////////////////////////////////////////////////////
//! Perform final stream compaction before writing out data
////////////////////////////////////////////////////////////////////////////////
__device__
void
compactStreamsFinal(const unsigned int tid, const unsigned int tid_2,
                    const unsigned int num_threads_active,
                    unsigned int &offset_mult_lambda,
                    float *s_left, float *s_right,
                    unsigned short *s_left_count, unsigned short *s_right_count,
                    unsigned short *s_cl_one, unsigned short *s_cl_mult,
                    unsigned short *s_cl_blocking, unsigned short *s_cl_helper,
                    unsigned int is_one_lambda, unsigned int is_one_lambda_2,
                    float &left, float &right, float &left_2, float &right_2,
                    unsigned int &left_count, unsigned int &right_count,
                    unsigned int &left_count_2, unsigned int &right_count_2,
                    unsigned int c_block_iend, unsigned int c_sum_block,
                    unsigned int c_block_iend_2, unsigned int c_sum_block_2
                   );

////////////////////////////////////////////////////////////////////////////////
//! Perform scan to compact list of block start addresses
////////////////////////////////////////////////////////////////////////////////
__device__
void
scanCompactBlocksStartAddress(const unsigned int tid, const unsigned int tid_2,
                              const unsigned int num_threads_compaction,
                              unsigned short *s_cl_blocking,
                              unsigned short *s_cl_helper
                             );

////////////////////////////////////////////////////////////////////////////////
//! Perform scan to obtain number of eigenvalues before a specific block
////////////////////////////////////////////////////////////////////////////////
__device__
void
scanSumBlocks(const unsigned int tid, const unsigned int tid_2,
              const unsigned int num_threads_active,
              const unsigned int num_threads_compaction,
              unsigned short *s_cl_blocking,
              unsigned short *s_cl_helper
             );

////////////////////////////////////////////////////////////////////////////////
//! Perform initial scan for compaction of intervals containing one and
//! multiple eigenvalues; also do initial scan to build blocks
////////////////////////////////////////////////////////////////////////////////
__device__
void
scanInitial(const unsigned int tid, const unsigned int tid_2,
            const unsigned int num_threads_active,
            const unsigned int num_threads_compaction,
            unsigned short *s_cl_one, unsigned short *s_cl_mult,
            unsigned short *s_cl_blocking, unsigned short *s_cl_helper
           );

////////////////////////////////////////////////////////////////////////////////
//! Store all non-empty intervals resulting from the subdivision of the interval
//! currently processed by the thread
//! @param  addr  address where to store
////////////////////////////////////////////////////////////////////////////////
__device__
void
storeNonEmptyIntervalsLarge(unsigned int addr,
                            const unsigned int num_threads_active,
                            float  *s_left, float *s_right,
                            unsigned short  *s_left_count,
                            unsigned short *s_right_count,
                            float left, float mid, float right,
                            const unsigned short left_count,
                            const unsigned short mid_count,
                            const unsigned short right_count,
                            float epsilon,
                            unsigned int &compact_second_chunk,
                            unsigned short *s_compaction_list,
                            unsigned int &is_active_second
                           );

////////////////////////////////////////////////////////////////////////////////
//! Bisection to find eigenvalues of a real, symmetric, and tridiagonal matrix
//! @param  g_d  diagonal elements in global memory
//! @param  g_s  superdiagonal elements in global elements (stored so that the
//!              element *(g_s - 1) can be accessed an equals 0
//! @param  n   size of matrix
//! @param  lg  lower bound of input interval (e.g. Gerschgorin interval)
//! @param  ug  upper bound of input interval (e.g. Gerschgorin interval)
//! @param  lg_eig_count  number of eigenvalues that are smaller than \a lg
//! @param  lu_eig_count  number of eigenvalues that are smaller than \a lu
//! @param  epsilon  desired accuracy of eigenvalues to compute
////////////////////////////////////////////////////////////////////////////////
__global__
void
bisectKernelLarge(float *g_d, float *g_s, const unsigned int n,
                  const float lg, const float ug,
                  const unsigned int lg_eig_count,
                  const unsigned int ug_eig_count,
                  float epsilon,
                  unsigned int *g_num_one,
                  unsigned int *g_num_blocks_mult,
                  float *g_left_one, float *g_right_one,
                  unsigned int *g_pos_one,
                  float *g_left_mult, float *g_right_mult,
                  unsigned int *g_left_count_mult,
                  unsigned int *g_right_count_mult,
                  unsigned int *g_blocks_mult,
                  unsigned int *g_blocks_mult_sum
                 )
{
    const unsigned int tid = threadIdx.x;

    // intervals (store left and right because the subdivision tree is in general
    // not dense
    __shared__  float  s_left[2 * MAX_THREADS_BLOCK + 1];
    __shared__  float  s_right[2 * MAX_THREADS_BLOCK + 1];

    // number of eigenvalues that are smaller than s_left / s_right
    // (correspondence is realized via indices)
    __shared__  unsigned short  s_left_count[2 * MAX_THREADS_BLOCK + 1];
    __shared__  unsigned short  s_right_count[2 * MAX_THREADS_BLOCK + 1];

    // helper for stream compaction
    __shared__  unsigned short  s_compaction_list[2 * MAX_THREADS_BLOCK + 1];

    // state variables for whole block
    // if 0 then compaction of second chunk of child intervals is not necessary
    // (because all intervals had exactly one non-dead child)
    __shared__  unsigned int compact_second_chunk;
    // if 1 then all threads are converged
    __shared__  unsigned int all_threads_converged;

    // number of currently active threads
    __shared__  unsigned int num_threads_active;

    // number of threads to use for stream compaction
    __shared__  unsigned int num_threads_compaction;

    // helper for exclusive scan
    unsigned short *s_compaction_list_exc = s_compaction_list + 1;


    // variables for currently processed interval
    // left and right limit of active interval
    float left = 0.0f;
    float right = 0.0f;
    unsigned int left_count = 0;
    unsigned int right_count = 0;
    // midpoint of active interval
    float  mid = 0.0f;
    // number of eigenvalues smaller then mid
    unsigned int mid_count = 0;
    // helper for stream compaction (tracking of threads generating second child)
    unsigned int is_active_second = 0;

    // initialize lists
    s_compaction_list[tid] = 0;
    s_left[tid] = 0;
    s_right[tid] = 0;
    s_left_count[tid] = 0;
    s_right_count[tid] = 0;

    __syncthreads();

    // set up initial configuration
    if (0 == tid)
    {

        s_left[0] = lg;
        s_right[0] = ug;
        s_left_count[0] = lg_eig_count;
        s_right_count[0] = ug_eig_count;

        compact_second_chunk = 0;
        num_threads_active = 1;

        num_threads_compaction = 1;

        all_threads_converged = 1;
    }

    __syncthreads();

    // for all active threads read intervals from the last level
    // the number of (worst case) active threads per level l is 2^l
    //while (true)                                                       
    for( unsigned int i = 0; i < 20; ++i )                                 // selbst hinzugefuegt
    {
        s_compaction_list[tid] = 0;
        s_compaction_list[tid + MAX_THREADS_BLOCK] = 0;
        s_compaction_list[2 * MAX_THREADS_BLOCK] = 0;
        subdivideActiveInterval(tid, s_left, s_right, s_left_count, s_right_count,
                                num_threads_active,
                                left, right, left_count, right_count,
                                mid, all_threads_converged);

        __syncthreads();

        // check if done
        if (1 == all_threads_converged)
        {
            break;
        }

        // compute number of eigenvalues smaller than mid
        // use all threads for reading the necessary matrix data from global
        // memory
        // use s_left and s_right as scratch space for diagonal and
        // superdiagonal of matrix
        mid_count = computeNumSmallerEigenvalsLarge(g_d, g_s, n,
                                                    mid, threadIdx.x,
                                                    num_threads_active,
                                                    s_left, s_right,
                                                    (left == right));

        __syncthreads();

        // store intervals
        // for all threads store the first child interval in a continuous chunk of
        // memory, and the second child interval -- if it exists -- in a second
        // chunk; it is likely that all threads reach convergence up to
        // \a epsilon at the same level; furthermore, for higher level most / all
        // threads will have only one child, storing the first child compactly will
        // (first) avoid to perform a compaction step on the first chunk, (second)
        // make it for higher levels (when all threads / intervals have
        // exactly one child)  unnecessary to perform a compaction of the second
        // chunk
        if (tid < num_threads_active)
        {

            if (left != right)
            {

                // store intervals
                storeNonEmptyIntervalsLarge(tid, num_threads_active,
                                            s_left, s_right,
                                            s_left_count, s_right_count,
                                            left, mid, right,
                                            left_count, mid_count, right_count,
                                            epsilon, compact_second_chunk,
                                            s_compaction_list_exc,
                                            is_active_second);
            }
            else
            {

                // re-write converged interval (has to be stored again because s_left
                // and s_right are used as scratch space for
                // computeNumSmallerEigenvalsLarge()
                s_left[tid] = left;
                s_right[tid] = left;
                s_left_count[tid] = left_count;
                s_right_count[tid] = right_count;

                is_active_second = 0;
            }
        }

        // necessary so that compact_second_chunk is up-to-date
        __syncthreads();

        // perform compaction of chunk where second children are stored
        // scan of (num_threads_active / 2) elements, thus at most
        // (num_threads_active / 4) threads are needed
        if (compact_second_chunk > 0)
        {

            // create indices for compaction
            createIndicesCompaction(s_compaction_list_exc, num_threads_compaction);
        }
        __syncthreads();
        
        if (compact_second_chunk > 0)                                              // selbst veraendert
        {
            compactIntervals(s_left, s_right, s_left_count, s_right_count,
                             mid, right, mid_count, right_count,
                             s_compaction_list, num_threads_active,
                             is_active_second);
        }

        __syncthreads();

        // update state variables
        if (0 == tid)
        {

            // update number of active threads with result of reduction
            num_threads_active += s_compaction_list[num_threads_active];
            num_threads_compaction = ceilPow2(num_threads_active);

            compact_second_chunk = 0;
            all_threads_converged = 1;
        }

        __syncthreads();

        if (num_threads_compaction > blockDim.x)
        {
            break;
        }

    }

    __syncthreads();

    // generate two lists of intervals; one with intervals that contain one
    // eigenvalue (or are converged), and one with intervals that need further
    // subdivision

    // perform two scans in parallel

    unsigned int left_count_2;
    unsigned int right_count_2;

    unsigned int tid_2 = tid + blockDim.x;

    // cache in per thread registers so that s_left_count and s_right_count
    // can be used for scans
    left_count = s_left_count[tid];
    right_count = s_right_count[tid];

    // some threads have to cache data for two intervals
    if (tid_2 < num_threads_active)
    {
        left_count_2 = s_left_count[tid_2];
        right_count_2 = s_right_count[tid_2];
    }

    // compaction list for intervals containing one and multiple eigenvalues
    // do not affect first element for exclusive scan
    unsigned short  *s_cl_one = s_left_count + 1;
    unsigned short  *s_cl_mult = s_right_count + 1;

    // compaction list for generating blocks of intervals containing multiple
    // eigenvalues
    unsigned short  *s_cl_blocking = s_compaction_list_exc;
    // helper compaction list for generating blocks of intervals
    __shared__ unsigned short  s_cl_helper[2 * MAX_THREADS_BLOCK + 1];

    if (0 == tid)
    {
        // set to 0 for exclusive scan
        s_left_count[0] = 0;
        s_right_count[0] = 0;
       
    }

    __syncthreads();

    // flag if interval contains one or multiple eigenvalues
    unsigned int is_one_lambda = 0;
    unsigned int is_one_lambda_2 = 0;

    // number of eigenvalues in the interval
    unsigned int multiplicity = right_count - left_count;
    is_one_lambda = (1 == multiplicity);

    s_cl_one[tid] = is_one_lambda;
    s_cl_mult[tid] = (! is_one_lambda);

    // (note: s_cl_blocking is non-zero only where s_cl_mult[] is non-zero)
    s_cl_blocking[tid] = (1 == is_one_lambda) ? 0 : multiplicity;
    s_cl_helper[tid] = 0;

    if (tid_2 < num_threads_active)
    {

        unsigned int multiplicity = right_count_2 - left_count_2;
        is_one_lambda_2 = (1 == multiplicity);

        s_cl_one[tid_2] = is_one_lambda_2;
        s_cl_mult[tid_2] = (! is_one_lambda_2);

        // (note: s_cl_blocking is non-zero only where s_cl_mult[] is non-zero)
        s_cl_blocking[tid_2] = (1 == is_one_lambda_2) ? 0 : multiplicity;
        s_cl_helper[tid_2] = 0;
    }
    else if (tid_2 < (2 * MAX_THREADS_BLOCK + 1))
    {

        // clear
        s_cl_blocking[tid_2] = 0;
        s_cl_helper[tid_2] = 0;
    }


    scanInitial(tid, tid_2, num_threads_active, num_threads_compaction,
                s_cl_one, s_cl_mult, s_cl_blocking, s_cl_helper);
    
    __syncthreads();                                                     // selbst hinzugefuegt

    scanSumBlocks(tid, tid_2, num_threads_active,
                  num_threads_compaction, s_cl_blocking, s_cl_helper);

    // end down sweep of scan
    __syncthreads();

    unsigned int  c_block_iend = 0;
    unsigned int  c_block_iend_2 = 0;
    unsigned int  c_sum_block = 0;
    unsigned int  c_sum_block_2 = 0;

    // for each thread / interval that corresponds to root node of interval block
    // store start address of block and total number of eigenvalues in all blocks
    // before this block (particular thread is irrelevant, constraint is to
    // have a subset of threads so that one and only one of them is in each
    // interval)
    if (1 == s_cl_helper[tid])
    {

        c_block_iend = s_cl_mult[tid] + 1;
        c_sum_block = s_cl_blocking[tid];
    }

    if (1 == s_cl_helper[tid_2])
    {

        c_block_iend_2 = s_cl_mult[tid_2] + 1;
        c_sum_block_2 = s_cl_blocking[tid_2];
    }

    scanCompactBlocksStartAddress(tid, tid_2, num_threads_compaction,
                                  s_cl_blocking, s_cl_helper);


    // finished second scan for s_cl_blocking
    __syncthreads();

    // determine the global results
    __shared__  unsigned int num_blocks_mult;
    __shared__  unsigned int num_mult;
    __shared__  unsigned int offset_mult_lambda;

    if (0 == tid)
    {

        num_blocks_mult = s_cl_blocking[num_threads_active - 1];
        offset_mult_lambda = s_cl_one[num_threads_active - 1];
        num_mult = s_cl_mult[num_threads_active - 1];

        *g_num_one = offset_mult_lambda;
        *g_num_blocks_mult = num_blocks_mult;
    }

    __syncthreads();

    float left_2, right_2;
    --s_cl_one;
    --s_cl_mult;
    --s_cl_blocking;
    
    //printf("s_l[%3u] = %10.8f\t s_r = %10.8f\t s_l_c = %3u\t s_r_c = %3u\n", 
      //tid, s_left[tid], s_right[tid], s_left_count[tid], s_right_count[tid]);
    printf("tid = %u\t left = %10.8f\t right = %10.8f\n", tid, left, right);
    
    __syncthreads();                                                        // selbst hinzugefuegt
    compactStreamsFinal(tid, tid_2, num_threads_active, offset_mult_lambda,
                        s_left, s_right, s_left_count, s_right_count,
                        s_cl_one, s_cl_mult, s_cl_blocking, s_cl_helper,
                        is_one_lambda, is_one_lambda_2,
                        left, right, left_2, right_2,
                        left_count, right_count, left_count_2, right_count_2,
                        c_block_iend, c_sum_block, c_block_iend_2, c_sum_block_2
                       );

    __syncthreads();

    // final adjustment before writing out data to global memory
    if (0 == tid)
    {
        s_cl_blocking[num_blocks_mult] = num_mult;
        s_cl_helper[0] = 0;
    }

    __syncthreads();

    // write to global memory
    writeToGmem(tid, tid_2, num_threads_active, num_blocks_mult,
                g_left_one, g_right_one, g_pos_one,
                g_left_mult, g_right_mult, g_left_count_mult, g_right_count_mult,
                s_left, s_right, s_left_count, s_right_count,
                g_blocks_mult, g_blocks_mult_sum,
                s_compaction_list, s_cl_helper, offset_mult_lambda);
                
}

////////////////////////////////////////////////////////////////////////////////
//! Write data to global memory
////////////////////////////////////////////////////////////////////////////////
__device__
void writeToGmem(const unsigned int tid, const unsigned int tid_2,
                 const unsigned int num_threads_active,
                 const unsigned int num_blocks_mult,
                 float *g_left_one, float *g_right_one,
                 unsigned int *g_pos_one,
                 float *g_left_mult, float *g_right_mult,
                 unsigned int *g_left_count_mult,
                 unsigned int *g_right_count_mult,
                 float *s_left, float *s_right,
                 unsigned short *s_left_count, unsigned short *s_right_count,
                 unsigned int *g_blocks_mult,
                 unsigned int *g_blocks_mult_sum,
                 unsigned short *s_compaction_list,
                 unsigned short *s_cl_helper,
                 unsigned int offset_mult_lambda
                )
{

    if (tid < offset_mult_lambda)
    {

        g_left_one[tid] = s_left[tid];
        g_right_one[tid] = s_right[tid];
        // right count can be used to order eigenvalues without sorting
        g_pos_one[tid] = s_right_count[tid];
    }
    else
    {

        
        g_left_mult[tid - offset_mult_lambda] = s_left[tid];
   /*     if (s_left[tid] == 2.5)
          printf("s_left[%u] = 2.5\n", tid);
        if (s_left[tid] == 2.5 && tid == 21)
          g_left_mult[tid - offset_mult_lambda] = 0.007074;*/
        g_right_mult[tid - offset_mult_lambda] = s_right[tid];
        g_left_count_mult[tid - offset_mult_lambda] = s_left_count[tid];
        g_right_count_mult[tid - offset_mult_lambda] = s_right_count[tid];
    }

    if (tid_2 < num_threads_active)
    {

        if (tid_2 < offset_mult_lambda)
        {

            g_left_one[tid_2] = s_left[tid_2];
            g_right_one[tid_2] = s_right[tid_2];
            // right count can be used to order eigenvalues without sorting
            g_pos_one[tid_2] = s_right_count[tid_2];
        }
        else
        {

            g_left_mult[tid_2 - offset_mult_lambda] = s_left[tid_2];
            g_right_mult[tid_2 - offset_mult_lambda] = s_right[tid_2];
            g_left_count_mult[tid_2 - offset_mult_lambda] = s_left_count[tid_2];
            g_right_count_mult[tid_2 - offset_mult_lambda] = s_right_count[tid_2];
        }

    } // end writing out data
    
    __syncthreads();                                                               // selbst hinzugefuegt

    // note that s_cl_blocking = s_compaction_list + 1;, that is by writing out
    // s_compaction_list we write the exclusive scan result
    if (tid <= num_blocks_mult)
    {
        g_blocks_mult[tid] = s_compaction_list[tid];
        g_blocks_mult_sum[tid] = s_cl_helper[tid];
    }

    if (tid_2 <= num_blocks_mult)
    {
        g_blocks_mult[tid_2] = s_compaction_list[tid_2];
        g_blocks_mult_sum[tid_2] = s_cl_helper[tid_2];
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Perform final stream compaction before writing data to global memory
////////////////////////////////////////////////////////////////////////////////
__device__
void
compactStreamsFinal(const unsigned int tid, const unsigned int tid_2,
                    const unsigned int num_threads_active,
                    unsigned int &offset_mult_lambda,
                    float *s_left, float *s_right,
                    unsigned short *s_left_count, unsigned short *s_right_count,
                    unsigned short *s_cl_one, unsigned short *s_cl_mult,
                    unsigned short *s_cl_blocking, unsigned short *s_cl_helper,
                    unsigned int is_one_lambda, unsigned int is_one_lambda_2,
                    float &left, float &right, float &left_2, float &right_2,
                    unsigned int &left_count, unsigned int &right_count,
                    unsigned int &left_count_2, unsigned int &right_count_2,
                    unsigned int c_block_iend, unsigned int c_sum_block,
                    unsigned int c_block_iend_2, unsigned int c_sum_block_2
                   )
{
    // cache data before performing compaction
   // left = s_left[tid];
  //  right = s_right[tid];

    if (tid_2 < num_threads_active)
    {

        left_2 = s_left[tid_2];
        right_2 = s_right[tid_2];
    }

    __syncthreads();

    // determine addresses for intervals containing multiple eigenvalues and
    // addresses for blocks of intervals
    unsigned int ptr_w = 0;
    unsigned int ptr_w_2 = 0;
    unsigned int ptr_blocking_w = 0;
    unsigned int ptr_blocking_w_2 = 0;
    
   

    ptr_w = (1 == is_one_lambda) ? s_cl_one[tid]
            : s_cl_mult[tid] + offset_mult_lambda;

    if (0 != c_block_iend)
    {
        ptr_blocking_w = s_cl_blocking[tid];
    }

    if (tid_2 < num_threads_active)
    {
        ptr_w_2 = (1 == is_one_lambda_2) ? s_cl_one[tid_2]
                  : s_cl_mult[tid_2] + offset_mult_lambda;

        if (0 != c_block_iend_2)
        {
            ptr_blocking_w_2 = s_cl_blocking[tid_2];
        }
    }
    
   
    __syncthreads();
    
    // store compactly in shared mem
      s_left[ptr_w] = left;
      s_right[ptr_w] = right;
      s_left_count[ptr_w] = left_count;
      s_right_count[ptr_w] = right_count;
    
    

    __syncthreads();                                                       // selbst hinzugefuegt
    if(tid == 1)
    {
      s_left[ptr_w] = left;
      s_right[ptr_w] = right;
      s_left_count[ptr_w] = left_count;
      s_right_count[ptr_w] = right_count;
      printf("s_l = %10.8f\t left = %10.8f\t s_r = %10.8f\t s_l_c = %u\t s_r_c = %u\t tid = %u \t ptr_w = %u\n",
        s_left[ptr_w], left, s_right[ptr_w], s_left_count[ptr_w], s_right_count[ptr_w], tid, ptr_w);

    
    }
          if (0 != c_block_iend)
    {
        s_cl_blocking[ptr_blocking_w + 1] = c_block_iend - 1;
        s_cl_helper[ptr_blocking_w + 1] = c_sum_block;
    }
    
  //  __syncthreads();                                                    // selbst hinzugefuegt
    if (tid_2 < num_threads_active)
    {

        // store compactly in shared mem
        s_left[ptr_w_2] = left_2;
        s_right[ptr_w_2] = right_2;
        s_left_count[ptr_w_2] = left_count_2;
        s_right_count[ptr_w_2] = right_count_2;

        if (0 != c_block_iend_2)
        {
            s_cl_blocking[ptr_blocking_w_2 + 1] = c_block_iend_2 - 1;
            s_cl_helper[ptr_blocking_w_2 + 1] = c_sum_block_2;
        }
    }

}

////////////////////////////////////////////////////////////////////////////////
//! Compute addresses to obtain compact list of block start addresses
////////////////////////////////////////////////////////////////////////////////
__device__
void
scanCompactBlocksStartAddress(const unsigned int tid, const unsigned int tid_2,
                              const unsigned int num_threads_compaction,
                              unsigned short *s_cl_blocking,
                              unsigned short *s_cl_helper
                             )
{
    // prepare for second step of block generation: compaction of the block
    // list itself to efficiently write out these
    s_cl_blocking[tid] = s_cl_helper[tid];

    if (tid_2 < num_threads_compaction)
    {
        s_cl_blocking[tid_2] = s_cl_helper[tid_2];
    }

    __syncthreads();

    // additional scan to compact s_cl_blocking that permits to generate a
    // compact list of eigenvalue blocks each one containing about
    // MAX_THREADS_BLOCK eigenvalues (so that each of these blocks may be
    // processed by one thread block in a subsequent processing step

    unsigned int offset = 1;

    // build scan tree
    for (int d = (num_threads_compaction >> 1); d > 0; d >>= 1)
    {

        __syncthreads();

        if (tid < d)
        {

            unsigned int  ai = offset*(2*tid+1)-1;
            unsigned int  bi = offset*(2*tid+2)-1;
            s_cl_blocking[bi] = s_cl_blocking[bi] + s_cl_blocking[ai];
        }

        offset <<= 1;
    }

    // traverse down tree: first down to level 2 across
    for (int d = 2; d < num_threads_compaction; d <<= 1)
    {

        offset >>= 1;
        __syncthreads();

        //
        if (tid < (d-1))
        {

            unsigned int  ai = offset*(tid+1) - 1;
            unsigned int  bi = ai + (offset >> 1);
            s_cl_blocking[bi] = s_cl_blocking[bi] + s_cl_blocking[ai];
        }
    }

}

////////////////////////////////////////////////////////////////////////////////
//! Perform scan to obtain number of eigenvalues before a specific block
////////////////////////////////////////////////////////////////////////////////
__device__
void
scanSumBlocks(const unsigned int tid, const unsigned int tid_2,
              const unsigned int num_threads_active,
              const unsigned int num_threads_compaction,
              unsigned short *s_cl_blocking,
              unsigned short *s_cl_helper)
{
    unsigned int offset = 1;

    // first step of scan to build the sum of elements within each block
    // build up tree
    for (int d = num_threads_compaction >> 1; d > 0; d >>= 1)
    {

        __syncthreads();

        if (tid < d)
        {

            unsigned int ai = offset*(2*tid+1)-1;
            unsigned int bi = offset*(2*tid+2)-1;

            s_cl_blocking[bi] += s_cl_blocking[ai];
        }

        offset *= 2;
    }

    // first step of scan to build the sum of elements within each block
    // traverse down tree
    for (int d = 2; d < (num_threads_compaction - 1); d <<= 1)
    {

        offset >>= 1;
        __syncthreads();

        if (tid < (d-1))
        {

            unsigned int ai = offset*(tid+1) - 1;
            unsigned int bi = ai + (offset >> 1);

            s_cl_blocking[bi] += s_cl_blocking[ai];
        }
    }

    __syncthreads();

    if (0 == tid)
    {

        // move last element of scan to last element that is valid
        // necessary because the number of threads employed for scan is a power
        // of two and not necessarily the number of active threasd
        s_cl_helper[num_threads_active - 1] =
            s_cl_helper[num_threads_compaction - 1];
        s_cl_blocking[num_threads_active - 1] =
            s_cl_blocking[num_threads_compaction - 1];
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Perform initial scan for compaction of intervals containing one and
//! multiple eigenvalues; also do initial scan to build blocks
////////////////////////////////////////////////////////////////////////////////
__device__
void
scanInitial(const unsigned int tid, const unsigned int tid_2,
            const unsigned int num_threads_active,
            const unsigned int num_threads_compaction,
            unsigned short *s_cl_one, unsigned short *s_cl_mult,
            unsigned short *s_cl_blocking, unsigned short *s_cl_helper
           )
{

    // perform scan to compactly write out the intervals containing one and
    // multiple eigenvalues
    // also generate tree for blocking of intervals containing multiple
    // eigenvalues

    unsigned int offset = 1;

    // build scan tree
    for (int d = (num_threads_compaction >> 1); d > 0; d >>= 1)
    {

        __syncthreads();

        if (tid < d)
        {

            unsigned int  ai = offset*(2*tid+1);
            unsigned int  bi = offset*(2*tid+2)-1;

            s_cl_one[bi] = s_cl_one[bi] + s_cl_one[ai - 1];
            s_cl_mult[bi] = s_cl_mult[bi] + s_cl_mult[ai - 1];

            // s_cl_helper is binary and zero for an internal node and 1 for a
            // root node of a tree corresponding to a block
            // s_cl_blocking contains the number of nodes in each sub-tree at each
            // iteration, the data has to be kept to compute the total number of
            // eigenvalues per block that, in turn, is needed to efficiently
            // write out data in the second step
            if ((s_cl_helper[ai - 1] != 1) || (s_cl_helper[bi] != 1))
            {

                // check how many childs are non terminated
                if (s_cl_helper[ai - 1] == 1)
                {
                    // mark as terminated
                    s_cl_helper[bi] = 1;
                }
                else if (s_cl_helper[bi] == 1)
                {
                    // mark as terminated
                    s_cl_helper[ai - 1] = 1;
                }
                else    // both childs are non-terminated
                {

                    unsigned int temp = s_cl_blocking[bi] + s_cl_blocking[ai - 1];

                    if (temp > MAX_THREADS_BLOCK)
                    {

                        // the two child trees have to form separate blocks, terminate trees
                        s_cl_helper[ai - 1] = 1;
                        s_cl_helper[bi] = 1;
                    }
                    else
                    {
                        // build up tree by joining subtrees
                        s_cl_blocking[bi] = temp;
                        s_cl_blocking[ai - 1] = 0;
                    }
                }
            }  // end s_cl_helper update

        }

        offset <<= 1;
    }


    // traverse down tree, this only for stream compaction, not for block
    // construction
    for (int d = 2; d < num_threads_compaction; d <<= 1)
    {

        offset >>= 1;
        __syncthreads();

        //
        if (tid < (d-1))
        {

            unsigned int  ai = offset*(tid+1) - 1;
            unsigned int  bi = ai + (offset >> 1);

            s_cl_one[bi] = s_cl_one[bi] + s_cl_one[ai];
            s_cl_mult[bi] = s_cl_mult[bi] + s_cl_mult[ai];
        }
    }

}

////////////////////////////////////////////////////////////////////////////////
//! Store all non-empty intervals resulting from the subdivision of the interval
//! currently processed by the thread
////////////////////////////////////////////////////////////////////////////////
__device__
void
storeNonEmptyIntervalsLarge(unsigned int addr,
                            const unsigned int num_threads_active,
                            float  *s_left, float *s_right,
                            unsigned short  *s_left_count,
                            unsigned short *s_right_count,
                            float left, float mid, float right,
                            const unsigned short left_count,
                            const unsigned short mid_count,
                            const unsigned short right_count,
                            float epsilon,
                            unsigned int &compact_second_chunk,
                            unsigned short *s_compaction_list,
                            unsigned int &is_active_second)
{
    // check if both child intervals are valid
    if ((left_count != mid_count) && (mid_count != right_count))
    {

        storeInterval(addr, s_left, s_right, s_left_count, s_right_count,
                      left, mid, left_count, mid_count, epsilon);

        is_active_second = 1;
        s_compaction_list[threadIdx.x] = 1;
        compact_second_chunk = 1;
    }
    else
    {

        // only one non-empty child interval

        // mark that no second child
        is_active_second = 0;
        s_compaction_list[threadIdx.x] = 0;

        // store the one valid child interval
        if (left_count != mid_count)
        {
            storeInterval(addr, s_left, s_right, s_left_count, s_right_count,
                          left, mid, left_count, mid_count, epsilon);
        }
        else
        {
            storeInterval(addr, s_left, s_right, s_left_count, s_right_count,
                          mid, right, mid_count, right_count, epsilon);
        }
    }
}

#endif // #ifndef _BISECT_KERNEL_LARGE_H_
