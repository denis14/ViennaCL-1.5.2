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

#ifndef _BISECT_KERNEL_LARGE_ONEI_H_
#define _BISECT_KERNEL_LARGE_ONEI_H_

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

void generate_bisectKernelLarge_OneIntervals(StringType & source, std::string const & numeric_string)
{
    source.append("     __kernel  \n");
    source.append("     void  \n");
    source.append("     bisectKernelLarge_OneIntervals(float *g_d, float *g_s, const unsigned int n,  \n");
    source.append("                                    unsigned int num_intervals,  \n");
    source.append("                                    float *g_left, float *g_right,  \n");
    source.append("                                    unsigned int *g_pos,  \n");
    source.append("                                    float  precision)  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");


    source.append("         const unsigned int gtid = (lcl_sz * grp_id) + lcl_id;  \n");

    source.append("         __local  float  s_left_scratch[MAX_THREADS_BLOCK];  \n");
    source.append("         __local  float  s_right_scratch[MAX_THREADS_BLOCK];  \n");

        // active interval of thread
        // left and right limit of current interval
    source.append("         float left, right;  \n");
        // number of threads smaller than the right limit (also corresponds to the
        // global index of the eigenvalues contained in the active interval)
    source.append("         unsigned int right_count;  \n");
        // flag if current thread converged
    source.append("         unsigned int converged = 0;  \n");
        // midpoint when current interval is subdivided
    source.append("         float mid = 0.0f;  \n");
        // number of eigenvalues less than mid
    source.append("         unsigned int mid_count = 0;  \n");

        // read data from global memory
    source.append("         if (gtid < num_intervals)  \n");
    source.append("         {  \n");
    source.append("             left = g_left[gtid];  \n");
    source.append("             right = g_right[gtid];  \n");
    source.append("             right_count = g_pos[gtid];  \n");
    source.append("         }  \n");


        // flag to determine if all threads converged to eigenvalue
    source.append("         __local  unsigned int  converged_all_threads;  \n");

        // initialized shared flag
    source.append("         if (0 == lcl_id)  \n");
    source.append("         {  \n");
    source.append("             converged_all_threads = 0;  \n");
    source.append("         }  \n");

    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

        // process until all threads converged to an eigenvalue
        // while( 0 == converged_all_threads) {
        //for (unsigned int i = 0; i < 5; ++i)                        // selbst hinzugefuegt
    source.append("         while (true)  \n");
    source.append("         {  \n");

    source.append("             converged_all_threads = 1;  \n");

            // update midpoint for all active threads
    source.append("             if ((gtid < num_intervals) && (0 == converged))  \n");
    source.append("             {  \n");

    source.append("                 mid = computeMidpoint(left, right);  \n");
    source.append("             }  \n");

            // find number of eigenvalues that are smaller than midpoint
    source.append("             mid_count = computeNumSmallerEigenvalsLarge(g_d, g_s, n,  \n");
    source.append("                                                         mid, gtid, num_intervals,  \n");
    source.append("                                                         s_left_scratch,  \n");
    source.append("                                                         s_right_scratch,  \n");
    source.append("                                                         converged);  \n");

    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

            // for all active threads
    source.append("             if ((gtid < num_intervals) && (0 == converged))  \n");
    source.append("             {  \n");

                // udpate intervals -- always one child interval survives
    source.append("                 if (right_count == mid_count)  \n");
    source.append("                 {  \n");
    source.append("                     right = mid;  \n");
    source.append("                 }  \n");
    source.append("                 else  \n");
    source.append("                 {  \n");
    source.append("                     left = mid;  \n");
    source.append("                 }  \n");

                // check for convergence
    source.append("                 float t0 = right - left;  \n");
    source.append("                 float t1 = max(abs(right), abs(left)) * precision;  \n");

    source.append("                 if (t0 < min(precision, t1))  \n");
    source.append("                 {  \n");

    source.append("                     float lambda = computeMidpoint(left, right);  \n");
    source.append("                     left = lambda;  \n");
    source.append("                     right = lambda;  \n");

    source.append("                     converged = 1;  \n");
    source.append("                 }  \n");
    source.append("                 else  \n");
    source.append("                 {  \n");
    source.append("                     converged_all_threads = 0;  \n");
    source.append("                 }  \n");
    source.append("             }  \n");

    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

    source.append("             if (1 == converged_all_threads)  \n");
    source.append("             {  \n");
    source.append("                 break;  \n");
    source.append("             }  \n");

    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
    source.append("         }  \n");

        // write data back to global memory
    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

    source.append("         if (gtid < num_intervals)  \n");
    source.append("         {  \n");
            // intervals converged so left and right interval limit are both identical
            // and identical to the eigenvalue
    source.append("             g_left[gtid] = left;  \n");
    source.append("         }  \n");
    source.append("     }  \n");
} 
 // }
//}
#endif // #ifndef _BISECT_KERNEL_LARGE_ONEI_H_




 
