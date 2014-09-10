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

/* Determine eigenvalues for small symmetric, tridiagonal matrix */

#ifndef _BISECT_KERNEL_SMALL_H_
#define _BISECT_KERNEL_SMALL_H_

// includes, project
#include "config.hpp"
#include "util.hpp"
#include "viennacl/vector.hpp"


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
#include "viennacl/linalg/host_based/matrix_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/...."  //<---------------------------------which file?
#endif

// additional kernel
#include "bisect_util.cpp"

namespace viennacl
{
    namespace linalg
    {

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
    ///
    template <typename T>
    void generate_bisect_kernel_small_bisectKernel(StringType & source, std::string const & numeric_string)
    {
        source.append("     __kernel  \n");
        source.append("     void  \n");
        source.append("     bisectKernel(float *g_d, float *g_s, const unsigned int n,  \n");
        source.append("                  __global "); source.append(numeric_string); source.append("* g_left, float *g_right,  \n");
        source.append("                  unsigned int *g_left_count, unsigned int *g_right_count,  \n");
        source.append("                  const float lg, const float ug,  \n");
        source.append("                  const unsigned int lg_eig_count, const unsigned int ug_eig_count,  \n");
        source.append("                  float epsilon  \n");
        source.append("                 )  \n");
        source.append("     {  \n");
        source.append("         uint glb_id = get_global_id(0); \n");
        source.append("         uint grp_id = get_group_id(0); \n");
        source.append("         uint grp_nm = get_num_groups(0); \n");
        source.append("         uint lcl_id = get_local_id(0); \n");
        source.append("         uint lcl_sz = get_local_size(0); \n");

            // intervals (store left and right because the subdivision tree is in general
            // not dense
        source.append("         __local  float  s_left[MAX_THREADS_BLOCK_SMALL_MATRIX];  \n");
        source.append("         __local  float  s_right[MAX_THREADS_BLOCK_SMALL_MATRIX];  \n");

            // number of eigenvalues that are smaller than s_left / s_right
            // (correspondence is realized via indices)
        source.append("         __local  unsigned int  s_left_count[MAX_THREADS_BLOCK_SMALL_MATRIX];  \n");
        source.append("         __local  unsigned int  s_right_count[MAX_THREADS_BLOCK_SMALL_MATRIX];  \n");

            // helper for stream compaction
        source.append("         __local  unsigned int  \n");
        source.append("         s_compaction_list[MAX_THREADS_BLOCK_SMALL_MATRIX + 1];  \n");

            // state variables for whole block
            // if 0 then compaction of second chunk of child intervals is not necessary
            // (because all intervals had exactly one non-dead child)
        source.append("         __local  unsigned int compact_second_chunk;  \n");
        source.append("         __local  unsigned int all_threads_converged;  \n");

            // number of currently active threads
        source.append("         __local  unsigned int num_threads_active;  \n");

            // number of threads to use for stream compaction
        source.append("         __local  unsigned int num_threads_compaction;  \n");

            // helper for exclusive scan
        source.append("         unsigned int *s_compaction_list_exc = s_compaction_list + 1;  \n");


            // variables for currently processed interval
            // left and right limit of active interval
        source.append("         float  left = 0.0f;  \n");
        source.append("         float  right = 0.0f;  \n");
        source.append("         unsigned int left_count = 0;  \n");
        source.append("         unsigned int right_count = 0;  \n");
            // midpoint of active interval
        source.append("         float  mid = 0.0f;  \n");
            // number of eigenvalues smaller then mid
        source.append("         unsigned int mid_count = 0;  \n");
            // affected from compaction
        source.append("         unsigned int  is_active_second = 0;  \n");

        source.append("         s_compaction_list[lcl_id] = 0;  \n");
        source.append("         s_left[lcl_id] = 0;  \n");
        source.append("         s_right[lcl_id] = 0;  \n");
        source.append("         s_left_count[lcl_id] = 0;  \n");
        source.append("         s_right_count[lcl_id] = 0;  \n");

        source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

            // set up initial configuration
        source.append("         if (0 == lcl_id)  \n");
        source.append("         {  \n");
        source.append("             s_left[0] = lg;  \n");
        source.append("             s_right[0] = ug;  \n");
        source.append("             s_left_count[0] = lg_eig_count;  \n");
        source.append("             s_right_count[0] = ug_eig_count;  \n");

        source.append("             compact_second_chunk = 0;  \n");
        source.append("             num_threads_active = 1;  \n");

        source.append("             num_threads_compaction = 1;  \n");
        source.append("         }  \n");

            // for all active threads read intervals from the last level
            // the number of (worst case) active threads per level l is 2^l
        source.append("         while (true)  \n");
        source.append("         {  \n");

        source.append("             all_threads_converged = 1;  \n");
        source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

        source.append("             is_active_second = 0;  \n");
        source.append("             subdivideActiveInterval(lcl_id,  \n");
        source.append("                                     s_left, s_right, s_left_count, s_right_count,  \n");
        source.append("                                     num_threads_active,  \n");
        source.append("                                     left, right, left_count, right_count,  \n");
        source.append("                                     mid, all_threads_converged);  \n");

        source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

                // check if done
        source.append("             if (1 == all_threads_converged)  \n");
        source.append("             {  \n");
        source.append("                 break;  \n");
        source.append("             }  \n");

        source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

                // compute number of eigenvalues smaller than mid
                // use all threads for reading the necessary matrix data from global
                // memory
                // use s_left and s_right as scratch space for diagonal and
                // superdiagonal of matrix
        source.append("             mid_count = computeNumSmallerEigenvals(g_d, g_s, n, mid,  \n");
        source.append("                                                    lcl_id, num_threads_active,  \n");
        source.append("                                                    s_left, s_right,  \n");
        source.append("                                                    (left == right));  \n");

        source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

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
        source.append("             if (lcl_id < num_threads_active)  \n");
        source.append("             {  \n");

        source.append("                 if (left != right)  \n");
        source.append("                 {  \n");

                        // store intervals
        source.append("                     storeNonEmptyIntervals(lcl_id, num_threads_active,  \n");
        source.append("                                            s_left, s_right, s_left_count, s_right_count,  \n");
        source.append("                                            left, mid, right,  \n");
        source.append("                                            left_count, mid_count, right_count,  \n");
        source.append("                                            epsilon, compact_second_chunk,  \n");
        source.append("                                            s_compaction_list_exc,  \n");
        source.append("                                            is_active_second);  \n");
        source.append("                 }  \n");
        source.append("                 else  \n");
        source.append("                 {  \n");

        source.append("                     storeIntervalConverged(s_left, s_right, s_left_count, s_right_count,  \n");
        source.append("                                            left, mid, right,  \n");
        source.append("                                            left_count, mid_count, right_count,  \n");
        source.append("                                            s_compaction_list_exc, compact_second_chunk,  \n");
        source.append("                                            num_threads_active,  \n");
        source.append("                                            is_active_second);  \n");
        source.append("                 }  \n");
        source.append("             }  \n");

                // necessary so that compact_second_chunk is up-to-date
        source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

                // perform compaction of chunk where second children are stored
                // scan of (num_threads_active / 2) elements, thus at most
                // (num_threads_active / 4) threads are needed
        source.append("             if (compact_second_chunk > 0)  \n");
        source.append("             {  \n");

        source.append("                 createIndicesCompaction(s_compaction_list_exc, num_threads_compaction);  \n");

        source.append("                 compactIntervals(s_left, s_right, s_left_count, s_right_count,  \n");
        source.append("                                  mid, right, mid_count, right_count,  \n");
        source.append("                                  s_compaction_list, num_threads_active,  \n");
        source.append("                                  is_active_second);  \n");
        source.append("             }  \n");

        source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

        source.append("             if (0 == lcl_id)  \n");
        source.append("             {  \n");

                    // update number of active threads with result of reduction
        source.append("                 num_threads_active += s_compaction_list[num_threads_active];  \n");

        source.append("                 num_threads_compaction = ceilPow2(num_threads_active);  \n");

        source.append("                 compact_second_chunk = 0;  \n");
        source.append("             }  \n");

        source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

        source.append("         }  \n");

        source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

            // write resulting intervals to global mem
            // for all threads write if they have been converged to an eigenvalue to
            // a separate array

            // at most n valid intervals
        source.append("         if (lcl_id < n)  \n");
        source.append("         {  \n");

                // intervals converged so left and right limit are identical
        source.append("             g_left[lcl_id]  = s_left[threadIdx.x];  \n");
                // left count is sufficient to have global order
        source.append("             g_left_count[lcl_id]  = s_left_count[threadIdx.x];  \n");
        source.append("         }  \n");
        source.append("     }  \n");
        source.append("   }  \n");
        source.append(" }  \n");
       }
#endif // #ifndef _BISECT_KERNEL_SMALL_H_




 
