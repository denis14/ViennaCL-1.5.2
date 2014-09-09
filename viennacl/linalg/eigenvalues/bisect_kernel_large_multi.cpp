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

//namespace viennacl
//{
  //namespace linalg
  //{
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

void generate_bisectKernelLarge_MultIntervals(StringType & source, std::string const & numeric_string)
{
    source.append("     __kernel  \n");
    source.append("     void  \n");
    source.append("     bisectKernelLarge_MultIntervals(float *g_d, float *g_s, const unsigned int n,  \n");
    source.append("                                     unsigned int *blocks_mult,  \n");
    source.append("                                     unsigned int *blocks_mult_sum,  \n");
    source.append("                                     float *g_left, float *g_right,  \n");
    source.append("                                     unsigned int *g_left_count,  \n");
    source.append("                                     unsigned int *g_right_count,  \n");
    source.append("                                     float *g_lambda, unsigned int *g_pos,  \n");
    source.append("                                     float precision  \n");
    source.append("                                    )  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");

    source.append("       const unsigned int tid = lcl_id;  \n");

        // left and right limits of interval
    source.append("         __local  float  s_left[2 * MAX_THREADS_BLOCK];  \n");
    source.append("         __local  float  s_right[2 * MAX_THREADS_BLOCK];  \n");

        // number of eigenvalues smaller than interval limits
    source.append("         __local  unsigned int  s_left_count[2 * MAX_THREADS_BLOCK];  \n");
    source.append("         __local  unsigned int  s_right_count[2 * MAX_THREADS_BLOCK];  \n");

        // helper array for chunk compaction of second chunk
    source.append("         __local  unsigned int  s_compaction_list[2 * MAX_THREADS_BLOCK + 1];  \n");
        // compaction list helper for exclusive scan
    source.append("         unsigned int *s_compaction_list_exc = s_compaction_list + 1;  \n");

        // flag if all threads are converged
    source.append("         __local  unsigned int  all_threads_converged;  \n");
        // number of active threads
    source.append("         __local  unsigned int  num_threads_active;  \n");
        // number of threads to employ for compaction
    source.append("         __local  unsigned int  num_threads_compaction;  \n");
        // flag if second chunk has to be compacted
    source.append("         __local  unsigned int compact_second_chunk;  \n");

        // parameters of block of intervals processed by this block of threads
    source.append("         __local  unsigned int  c_block_start;  \n");
    source.append("         __local  unsigned int  c_block_end;  \n");
    source.append("         __local  unsigned int  c_block_offset_output;  \n");

        // midpoint of currently active interval of the thread
    source.append("         float mid = 0.0f;  \n");
        // number of eigenvalues smaller than \a mid
    source.append("         unsigned int  mid_count = 0;  \n");
        // current interval parameter
    source.append("         float  left = 0.0f;  \n");
    source.append("         float  right = 0.0f;  \n");
    source.append("         unsigned int  left_count = 0;  \n");
    source.append("         unsigned int  right_count = 0;  \n");
        // helper for compaction, keep track which threads have a second child
    source.append("         unsigned int  is_active_second = 0;  \n");

// selbst hinzugefuegt
    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;                                        \n");

        // initialize common start conditions
    source.append("         if (0 == tid)  \n");
    source.append("         {  \n");

    source.append("             c_block_start = blocks_mult[grp_id];  \n");
    source.append("             c_block_end = blocks_mult[grp_id + 1];  \n");
    source.append("             c_block_offset_output = blocks_mult_sum[grp_id];  \n");
    source.append("               \n");

    source.append("             num_threads_active = c_block_end - c_block_start;  \n");
    source.append("             s_compaction_list[0] = 0;  \n");
    source.append("             num_threads_compaction = ceilPow2(num_threads_active);  \n");

    source.append("             all_threads_converged = 1;  \n");
    source.append("             compact_second_chunk = 0;  \n");
    source.append("         }  \n");
   // selbst hinzugefuegt
    source.append("          s_left_count [tid] = 42;                                        \n");
    source.append("          s_right_count[tid] = 42;  \n");
    source.append("          s_left_count [tid + MAX_THREADS_BLOCK] = 0;  \n");
    source.append("          s_right_count[tid + MAX_THREADS_BLOCK] = 0;  \n");
    source.append("           \n");
    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
    source.append("           \n");

        // read data into shared memory
    source.append("         if (tid < num_threads_active)  \n");
    source.append("         {  \n");

    source.append("             s_left[tid]  = g_left[c_block_start + tid];  \n");
    source.append("             s_right[tid] = g_right[c_block_start + tid];  \n");
    source.append("             s_left_count[tid]  = g_left_count[c_block_start + tid];  \n");
    source.append("             s_right_count[tid] = g_right_count[c_block_start + tid];  \n");
           // printf("1: tid = %u s_l = %10.8f \t s_r = %10.8f \t s_l_c = %u \t s_r_c = %u \t c_block_start + tid = %u\n", 
             // tid, s_left[tid], s_right[tid], s_left_count[tid], s_right_count[tid], c_block_start + tid);       // selbst hinzugefuegt

    source.append("               \n");
    source.append("         }  \n");
    source.append("        \n");
    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
    source.append("         unsigned int iter = 0;  \n");
        // do until all threads converged
    source.append("         while (true)  \n");
    source.append("         {  \n");
    source.append("             iter++;  \n");
            //for (int iter=0; iter < 0; iter++) {
    source.append("             s_compaction_list[lcl_id] = 0;  \n");
    source.append("             s_compaction_list[lcl_id + lcl_sz] = 0;  \n");
    source.append("             s_compaction_list[2 * MAX_THREADS_BLOCK] = 0;  \n");

            // subdivide interval if currently active and not already converged
    source.append("             subdivideActiveInterval(tid, s_left, s_right,  \n");
    source.append("                                     s_left_count, s_right_count,  \n");
    source.append("                                     num_threads_active,  \n");
    source.append("                                     left, right, left_count, right_count,  \n");
    source.append("                                     mid, all_threads_converged);  \n");
    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

            // stop if all eigenvalues have been found
    source.append("             if (1 == all_threads_converged)  \n");
    source.append("             {  \n");
    source.append("                  \n");
    source.append("                 break;  \n");
    source.append("             }  \n");

            // compute number of eigenvalues smaller than mid for active and not
            // converged intervals, use all threads for loading data from gmem and
            // s_left and s_right as scratch space to store the data load from gmem
            // in shared memory
    source.append("             mid_count = computeNumSmallerEigenvalsLarge(g_d, g_s, n,  \n");
    source.append("                                                         mid, tid, num_threads_active,  \n");
    source.append("                                                         s_left, s_right,  \n");
    source.append("                                                         (left == right));  \n");
    source.append("                                                \n");
    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

    source.append("             if (tid < num_threads_active)  \n");
    source.append("             {  \n");
    source.append("                   \n");
                // store intervals
    source.append("                 if (left != right)  \n");
    source.append("                 {  \n");

    source.append("                     storeNonEmptyIntervals(tid, num_threads_active,  \n");
    source.append("                                            s_left, s_right, s_left_count, s_right_count,  \n");
    source.append("                                            left, mid, right,  \n");
    source.append("                                            left_count, mid_count, right_count,  \n");
    source.append("                                            precision, compact_second_chunk,  \n");
    source.append("                                            s_compaction_list_exc,  \n");
    source.append("                                            is_active_second);  \n");
    source.append("                      \n");
    source.append("                 }  \n");
    source.append("                 else  \n");
    source.append("                 {  \n");

    source.append("                     storeIntervalConverged(s_left, s_right, s_left_count, s_right_count,  \n");
    source.append("                                            left, mid, right,  \n");
    source.append("                                            left_count, mid_count, right_count,  \n");
    source.append("                                            s_compaction_list_exc, compact_second_chunk,  \n");
    source.append("                                            num_threads_active,  \n");
    source.append("                                            is_active_second);  \n");
    source.append("                   \n");
    source.append("                 }  \n");
    source.append("             }  \n");

    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

            // compact second chunk of intervals if any of the threads generated
            // two child intervals
    source.append("             if (1 == compact_second_chunk)  \n");
    source.append("             {  \n");

    source.append("                 createIndicesCompaction(s_compaction_list_exc, num_threads_compaction);  \n");
    source.append("                 compactIntervals(s_left, s_right, s_left_count, s_right_count,  \n");
    source.append("                                  mid, right, mid_count, right_count,  \n");
    source.append("                                  s_compaction_list, num_threads_active,  \n");
    source.append("                                  is_active_second);  \n");
    source.append("             }  \n");

    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

            // update state variables
    source.append("             if (0 == tid)  \n");
    source.append("             {  \n");
    source.append("                 num_threads_active += s_compaction_list[num_threads_active];  \n");
    source.append("                 num_threads_compaction = ceilPow2(num_threads_active);  \n");

    source.append("                 compact_second_chunk = 0;  \n");
    source.append("                 all_threads_converged = 1;  \n");
    source.append("             }  \n");

    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

            // clear
    source.append("             s_compaction_list_exc[lcl_id] = 0;  \n");
		 // selbst hinzugefuegt
    source.append("             s_compaction_list_exc[lcl_id + lcl_sz] = 0;   \n");
    source.append("               \n");
    source.append("             if (num_threads_compaction > lcl_sz)              \n");
    source.append("             {  \n");
    source.append("               break;  \n");
    source.append("             }  \n");


    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

    source.append("    } \n"); // end until all threads converged

        // write data back to global memory
    source.append("         if (tid < num_threads_active)  \n");
    source.append("         {  \n");

    source.append("             unsigned int addr = c_block_offset_output + tid;  \n");
    source.append("               \n");
    source.append("             g_lambda[addr]  = s_left[tid];  \n");
    source.append("             g_pos[addr]   = s_right_count[tid];  \n");
            //printf("s_left = %10.8f\n", s_left[tid]);
    source.append("         }  \n");
    source.append("     }  \n");
} 
  //}
//}

#endif // #ifndef _BISECT_KERNEL_LARGE_MULTI_H_




 
