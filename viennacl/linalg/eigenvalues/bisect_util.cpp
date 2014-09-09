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

/* Utility / shared functionality for bisection kernels */

#ifndef _BISECT_UTIL_H_
#define _BISECT_UTIL_H_

// includes, project
#include "config.hpp"
#include "util.hpp"

//namespace viennacl
//{
  //namespace linalg
  //{
    ////////////////////////////////////////////////////////////////////////////////
    //! Compute the next lower power of two of n
    //! @param  n  number for which next higher power of two is seeked
    ////////////////////////////////////////////////////////////////////////////////

void generate_floorPow2(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     inline int  \n");
    source.append("     floorPow2(int n)  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");


        // early out if already power of two
    source.append("         if (0 == (n & (n-1)))  \n");
    source.append("         {  \n");
    source.append("             return n;  \n");
    source.append("         }  \n");

    source.append("         int exp;  \n");
    source.append("         frexp((float)n, &exp);  \n");
    source.append("         return (1 << (exp - 1));  \n");
    source.append("     }  \n");

} 


////////////////////////////////////////////////////////////////////////////////
//! Compute the next higher power of two of n
//! @param  n  number for which next higher power of two is seeked
////////////////////////////////////////////////////////////////////////////////

void generate_ceilPow2(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     inline int  \n");
    source.append("     ceilPow2(int n)  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");


        // early out if already power of two
    source.append("         if (0 == (n & (n-1)))  \n");
    source.append("         {  \n");
    source.append("             return n;  \n");
    source.append("         }  \n");

    source.append("         int exp;  \n");
    source.append("         frexp((float)n, &exp);  \n");
    source.append("         return (1 << exp);  \n");
    source.append("     }  \n");
} 


////////////////////////////////////////////////////////////////////////////////
//! Compute midpoint of interval [\a left, \a right] avoiding overflow if
//! possible
//! @param left   left / lower limit of interval
//! @param right  right / upper limit of interval
////////////////////////////////////////////////////////////////////////////////

void generate_computeMidpoint(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     inline float  \n");
    source.append("     computeMidpoint(const float left, const float right)  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");


    source.append("         float mid;  \n");

    source.append("         if (sign_f(left) == sign_f(right))  \n");
    source.append("         {  \n");
    source.append("             mid = left + (right - left) * 0.5f;  \n");
    source.append("         }  \n");
    source.append("         else  \n");
    source.append("         {  \n");
    source.append("             mid = (left + right) * 0.5f;  \n");
    source.append("         }  \n");

    source.append("         return mid;  \n");
    source.append("     }  \n");

} 


////////////////////////////////////////////////////////////////////////////////
//! Check if interval converged and store appropriately
//! @param  addr    address where to store the information of the interval
//! @param  s_left  shared memory storage for left interval limits
//! @param  s_right  shared memory storage for right interval limits
//! @param  s_left_count  shared memory storage for number of eigenvalues less
//!                       than left interval limits
//! @param  s_right_count  shared memory storage for number of eigenvalues less
//!                       than right interval limits
//! @param  left   lower limit of interval
//! @param  right  upper limit of interval
//! @param  left_count  eigenvalues less than \a left
//! @param  right_count  eigenvalues less than \a right
//! @param  precision  desired precision for eigenvalues
////////////////////////////////////////////////////////////////////////////////

template<class S, class T>
void generate_storeInterval(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     storeInterval(unsigned int addr,  \n");
    source.append("                   float *s_left, float *s_right,  \n");
    source.append("                   __global "); source.append(numeric_string); source.append("*s_left_count, T *s_right_count,  \n");
    source.append("                   float left, float right,  \n");
    source.append("                   S left_count, S right_count,  \n");
    source.append("                   float precision)  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");

    source.append("         s_left_count[addr] = left_count;  \n");
    source.append("         s_right_count[addr] = right_count;  \n");

        // check if interval converged
    source.append("         float t0 = abs(right - left);  \n");
    source.append("         float t1 = max(abs(left), abs(right)) * precision;  \n");

    source.append("         if (t0 <= max((float)MIN_ABS_INTERVAL, t1))  \n");
    source.append("         {  \n");
            // compute mid point
    source.append("             float lambda = computeMidpoint(left, right);  \n");

            // mark as converged
    source.append("             s_left[addr] = lambda;  \n");
    source.append("             s_right[addr] = lambda;  \n");
    source.append("         }  \n");
    source.append("         else  \n");
    source.append("         {  \n");

            // store current limits
    source.append("             s_left[addr] = left;  \n");
    source.append("             s_right[addr] = right;  \n");
    source.append("         }  \n");
    source.append("     }  \n");

} 



////////////////////////////////////////////////////////////////////////////////
//! Compute number of eigenvalues that are smaller than x given a symmetric,
//! real, and tridiagonal matrix
//! @param  g_d  diagonal elements stored in global memory
//! @param  g_s  superdiagonal elements stored in global memory
//! @param  n    size of matrix
//! @param  x    value for which the number of eigenvalues that are smaller is
//!              seeked
//! @param  tid  thread identified (e.g. threadIdx.x or gtid)
//! @param  num_intervals_active  number of active intervals / threads that
//!                               currently process an interval
//! @param  s_d  scratch space to store diagonal entries of the tridiagonal
//!              matrix in shared memory
//! @param  s_s  scratch space to store superdiagonal entries of the tridiagonal
//!              matrix in shared memory
//! @param  converged  flag if the current thread is already converged (that
//!         is count does not have to be computed)
////////////////////////////////////////////////////////////////////////////////

void generate_computeNumSmallerEigenvals(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     inline unsigned int  \n");
    source.append("     computeNumSmallerEigenvals(float *g_d, float *g_s, const unsigned int n,  \n");
    source.append("                                const float x,  \n");
    source.append("                                const unsigned int tid,  \n");
    source.append("                                const unsigned int num_intervals_active,  \n");
    source.append("                                float *s_d, float *s_s,  \n");
    source.append("                                unsigned int converged  \n");
    source.append("                               )  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");


    source.append("         float  delta = 1.0f;  \n");
    source.append("         unsigned int count = 0;  \n");

    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

        // read data into shared memory
    source.append("         if (lcl_id < n)  \n");
    source.append("         {  \n");
    source.append("             s_d[lcl_id] = *(g_d + threadIdx.x);  \n");
    source.append("             s_s[lcl_id] = *(g_s + threadIdx.x - 1);  \n");
    source.append("         }  \n");

    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

        // perform loop only for active threads
    source.append("         if ((tid < num_intervals_active) && (0 == converged))  \n");
    source.append("         {  \n");

            // perform (optimized) Gaussian elimination to determine the number
            // of eigenvalues that are smaller than n
    source.append("             for (unsigned int k = 0; k < n; ++k)  \n");
    source.append("             {  \n");
    source.append("                 delta = s_d[k] - x - (s_s[k] * s_s[k]) / delta;  \n");
    source.append("                 count += (delta < 0) ? 1 : 0;  \n");
    source.append("             }  \n");

    source.append("         } \n"); // end if thread currently processing an interval

    source.append("         return count;  \n");
    source.append("     }  \n");

} 


////////////////////////////////////////////////////////////////////////////////
//! Compute number of eigenvalues that are smaller than x given a symmetric,
//! real, and tridiagonal matrix
//! @param  g_d  diagonal elements stored in global memory
//! @param  g_s  superdiagonal elements stored in global memory
//! @param  n    size of matrix
//! @param  x    value for which the number of eigenvalues that are smaller is
//!              seeked
//! @param  tid  thread identified (e.g. threadIdx.x or gtid)
//! @param  num_intervals_active  number of active intervals / threads that
//!                               currently process an interval
//! @param  s_d  scratch space to store diagonal entries of the tridiagonal
//!              matrix in shared memory
//! @param  s_s  scratch space to store superdiagonal entries of the tridiagonal
//!              matrix in shared memory
//! @param  converged  flag if the current thread is already converged (that
//!         is count does not have to be computed)
////////////////////////////////////////////////////////////////////////////////

void generate_computeNumSmallerEigenvalsLarge(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     inline unsigned int  \n");
    source.append("     computeNumSmallerEigenvalsLarge(float *g_d, float *g_s, const unsigned int n,  \n");
    source.append("                                     const float x,  \n");
    source.append("                                     const unsigned int tid,  \n");
    source.append("                                     const unsigned int num_intervals_active,  \n");
    source.append("                                     float *s_d, float *s_s,  \n");
    source.append("                                     unsigned int converged  \n");
    source.append("                                    )  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");

    source.append("         float  delta = 1.0f;  \n");
    source.append("         unsigned int count = 0;  \n");

    source.append("         unsigned int rem = n;  \n");

        // do until whole diagonal and superdiagonal has been loaded and processed
    source.append("         for (unsigned int i = 0; i < n; i += lcl_sz)  \n");
    source.append("         {  \n");

    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

            // read new chunk of data into shared memory
    source.append("             if ((i + lcl_id) < n)  \n");
    source.append("             {  \n");

    source.append("                 s_d[lcl_id] = *(g_d + i + threadIdx.x);  \n");
    source.append("                 s_s[lcl_id] = *(g_s + i + threadIdx.x - 1);  \n");
    source.append("             }  \n");

    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");


    source.append("             if (tid < num_intervals_active)  \n");
    source.append("             {  \n");

                // perform (optimized) Gaussian elimination to determine the number
                // of eigenvalues that are smaller than n
    source.append("                 for (unsigned int k = 0; k < min(rem,lcl_sz); ++k)  \n");
    source.append("                 {  \n");
    source.append("                     delta = s_d[k] - x - (s_s[k] * s_s[k]) / delta;  \n");
                    // delta = (abs( delta) < (1.0e-10)) ? -(1.0e-10) : delta;
    source.append("                     count += (delta < 0) ? 1 : 0;  \n");
    source.append("                 }  \n");

    source.append("             } \n"); // end if thread currently processing an interval

    source.append("             rem -= lcl_sz;  \n");
    source.append("         }  \n");

    source.append("         return count;  \n");
    source.append("     }  \n");


} 

////////////////////////////////////////////////////////////////////////////////
//! Store all non-empty intervals resulting from the subdivision of the interval
//! currently processed by the thread
//! @param  addr  base address for storing intervals
//! @param  num_threads_active  number of threads / intervals in current sweep
//! @param  s_left  shared memory storage for left interval limits
//! @param  s_right  shared memory storage for right interval limits
//! @param  s_left_count  shared memory storage for number of eigenvalues less
//!                       than left interval limits
//! @param  s_right_count  shared memory storage for number of eigenvalues less
//!                       than right interval limits
//! @param  left   lower limit of interval
//! @param  mid    midpoint of interval
//! @param  right  upper limit of interval
//! @param  left_count  eigenvalues less than \a left
//! @param  mid_count  eigenvalues less than \a mid
//! @param  right_count  eigenvalues less than \a right
//! @param  precision  desired precision for eigenvalues
//! @param  compact_second_chunk  shared mem flag if second chunk is used and
//!                               ergo requires compaction
//! @param  s_compaction_list_exc  helper array for stream compaction,
//!                                s_compaction_list_exc[tid] = 1 when the
//!                                thread generated two child intervals
//! @is_active_interval  mark is thread has a second non-empty child interval
////////////////////////////////////////////////////////////////////////////////

template<class S, class T>
void generate_storeNonEmptyIntervals(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     storeNonEmptyIntervals(unsigned int addr,  \n");
    source.append("                            const unsigned int num_threads_active,  \n");
    source.append("                            float  *s_left, float *s_right,  \n");
    source.append("                            T  *s_left_count, __global "); source.append(numeric_string); source.append("*s_right_count,  \n");
    source.append("                            float left, float mid, float right,  \n");
    source.append("                            const S left_count,  \n");
    source.append("                            const S mid_count,  \n");
    source.append("                            const S right_count,  \n");
    source.append("                            float precision,  \n");
    source.append("                            unsigned int &compact_second_chunk,  \n");
    source.append("                            __global "); source.append(numeric_string); source.append("*s_compaction_list_exc,  \n");
    source.append("                            unsigned int &is_active_second)  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");

        // check if both child intervals are valid
    source.append("          \n");
    source.append("         if ((left_count != mid_count) && (mid_count != right_count))  \n");
    source.append("         {  \n");

            // store the left interval
    source.append("             storeInterval(addr, s_left, s_right, s_left_count, s_right_count,  \n");
    source.append("                           left, mid, left_count, mid_count, precision);  \n");

            // mark that a second interval has been generated, only stored after
            // stream compaction of second chunk
    source.append("             is_active_second = 1;  \n");
    source.append("             s_compaction_list_exc[lcl_id] = 1;  \n");
    source.append("             compact_second_chunk = 1;  \n");
    source.append("         }  \n");
    source.append("         else  \n");
    source.append("         {  \n");

            // only one non-empty child interval

            // mark that no second child
    source.append("             is_active_second = 0;  \n");
    source.append("             s_compaction_list_exc[lcl_id] = 0;  \n");

            // store the one valid child interval
    source.append("             if (left_count != mid_count)  \n");
    source.append("             {  \n");
    source.append("                 storeInterval(addr, s_left, s_right, s_left_count, s_right_count,  \n");
    source.append("                               left, mid, left_count, mid_count, precision);  \n");
    source.append("             }  \n");
    source.append("             else  \n");
    source.append("             {  \n");
    source.append("                 storeInterval(addr, s_left, s_right, s_left_count, s_right_count,  \n");
    source.append("                               mid, right, mid_count, right_count, precision);  \n");
    source.append("             }  \n");

    source.append("         }  \n");
    source.append("     }  \n");

} 

////////////////////////////////////////////////////////////////////////////////
//! Create indices for compaction, that is process \a s_compaction_list_exc
//! which is 1 for intervals that generated a second child and 0 otherwise
//! and create for each of the non-zero elements the index where the new
//! interval belongs to in a compact representation of all generated second
//! childs
//! @param   s_compaction_list_exc  list containing the flags which threads
//!                                 generated two childs
//! @param   num_threads_compaction number of threads to employ for compaction
////////////////////////////////////////////////////////////////////////////////

template<class T>
void generate_createIndicesCompaction(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     createIndicesCompaction(__global "); source.append(numeric_string); source.append("*s_compaction_list_exc,  \n");
    source.append("                             unsigned int num_threads_compaction)  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");


    source.append("         unsigned int offset = 1;  \n");
    source.append("         const unsigned int tid = lcl_id;  \n");
       // if(tid == 0)
         // printf("num_threads_compaction = %u\n", num_threads_compaction);

        // higher levels of scan tree
    source.append("         for (int d = (num_threads_compaction >> 1); d > 0; d >>= 1)  \n");
    source.append("         {  \n");

    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

    source.append("             if (tid < d)  \n");
    source.append("             {  \n");

    source.append("                 unsigned int  ai = offset*(2*tid+1)-1;  \n");
    source.append("                 unsigned int  bi = offset*(2*tid+2)-1;  \n");
    source.append("              \n");
    source.append("                 s_compaction_list_exc[bi] =   s_compaction_list_exc[bi]  \n");
    source.append("                                               + s_compaction_list_exc[ai];  \n");
    source.append("             }  \n");

    source.append("             offset <<= 1;  \n");
    source.append("         }  \n");

        // traverse down tree: first down to level 2 across
    source.append("         for (int d = 2; d < num_threads_compaction; d <<= 1)  \n");
    source.append("         {  \n");

    source.append("             offset >>= 1;  \n");
    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

    source.append("             if (tid < (d-1))  \n");
    source.append("             {  \n");

    source.append("                 unsigned int  ai = offset*(tid+1) - 1;  \n");
    source.append("                 unsigned int  bi = ai + (offset >> 1);  \n");

    source.append("                 s_compaction_list_exc[bi] =   s_compaction_list_exc[bi]  \n");
    source.append("                                               + s_compaction_list_exc[ai];  \n");
    source.append("             }  \n");
    source.append("         }  \n");

    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

    source.append("     }  \n");
} 

///////////////////////////////////////////////////////////////////////////////
//! Perform stream compaction for second child intervals
//! @param  s_left  shared
//! @param  s_left  shared memory storage for left interval limits
//! @param  s_right  shared memory storage for right interval limits
//! @param  s_left_count  shared memory storage for number of eigenvalues less
//!                       than left interval limits
//! @param  s_right_count  shared memory storage for number of eigenvalues less
//!                       than right interval limits
//! @param  mid    midpoint of current interval (left of new interval)
//! @param  right  upper limit of interval
//! @param  mid_count  eigenvalues less than \a mid
//! @param  s_compaction_list  list containing the indices where the data has
//!         to be stored
//! @param  num_threads_active  number of active threads / intervals
//! @is_active_interval  mark is thread has a second non-empty child interval
///////////////////////////////////////////////////////////////////////////////

 
template<class T>
void generate_compactIntervals(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     compactIntervals(float *s_left, float *s_right,  \n");
    source.append("                      __global "); source.append(numeric_string); source.append("*s_left_count, T *s_right_count,  \n");
    source.append("                      float mid, float right,  \n");
    source.append("                      unsigned int mid_count, unsigned int right_count,  \n");
    source.append("                      __global "); source.append(numeric_string); source.append("*s_compaction_list,  \n");
    source.append("                      unsigned int num_threads_active,  \n");
    source.append("                      unsigned int is_active_second)  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");

    source.append("         const unsigned int tid = lcl_id;  \n");

        // perform compaction / copy data for all threads where the second
        // child is not dead
    source.append("         if ((tid < num_threads_active) && (1 == is_active_second))  \n");
    source.append("         {  \n");
    source.append("             unsigned int addr_w = num_threads_active + s_compaction_list[tid];  \n");
    source.append("             s_left[addr_w] = mid;  \n");
    source.append("             s_right[addr_w] = right;  \n");
    source.append("             s_left_count[addr_w] = mid_count;  \n");
    source.append("             s_right_count[addr_w] = right_count;  \n");
    source.append("         }  \n");
    source.append("     }  \n");
} 


 
template<class T, class S>
void generate_storeIntervalConverged(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     storeIntervalConverged(float *s_left, float *s_right,  \n");
    source.append("                            __global "); source.append(numeric_string); source.append("*s_left_count, T *s_right_count,  \n");
    source.append("                            float &left, float &mid, float &right,  \n");
    source.append("                            S &left_count, S &mid_count, S &right_count,  \n");
    source.append("                            __global "); source.append(numeric_string); source.append("*s_compaction_list_exc,  \n");
    source.append("                            unsigned int &compact_second_chunk,  \n");
    source.append("                            const unsigned int num_threads_active,  \n");
    source.append("                            unsigned int &is_active_second)  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");

    source.append("         const unsigned int tid = lcl_id;  \n");
       // const unsigned int multiplicity = right_count >= left_count ? right_count - left_count : 1; // selbst veraendert
    source.append("    const unsigned int multiplicity = right_count - left_count;  \n");  // selbst veraendert
        // check multiplicity of eigenvalue
    source.append("         if (1 == multiplicity)  \n");
    source.append("         {  \n");

            // just re-store intervals, simple eigenvalue
    source.append("             s_left[tid] = left;  \n");
    source.append("             s_right[tid] = right;  \n");
    source.append("             s_left_count[tid] = left_count;  \n");
    source.append("             s_right_count[tid] = right_count;  \n");
    source.append("             \n");

            // mark that no second child / clear
    source.append("             is_active_second = 0;  \n");
    source.append("             s_compaction_list_exc[tid] = 0;  \n");
    source.append("         }  \n");
    source.append("         else  \n");
    source.append("         {  \n");

            // number of eigenvalues after the split less than mid
    source.append("             mid_count = left_count + (multiplicity >> 1);  \n");

            // store left interval
    source.append("             s_left[tid] = left;  \n");
    source.append("             s_right[tid] = right;  \n");
    source.append("             s_left_count[tid] = left_count;  \n");
    source.append("             s_right_count[tid] = mid_count;  \n");
    source.append("             mid = left;  \n");

            // mark that second child interval exists
    source.append("             is_active_second = 1;  \n");
    source.append("             s_compaction_list_exc[tid] = 1;  \n");
    source.append("             compact_second_chunk = 1;  \n");
    source.append("         }  \n");
    source.append("     }  \n");
}

///////////////////////////////////////////////////////////////////////////////
//! Subdivide interval if active and not already converged
//! @param tid  id of thread
//! @param  s_left  shared memory storage for left interval limits
//! @param  s_right  shared memory storage for right interval limits
//! @param  s_left_count  shared memory storage for number of eigenvalues less
//!                       than left interval limits
//! @param  s_right_count  shared memory storage for number of eigenvalues less
//!                       than right interval limits
//! @param  num_threads_active  number of active threads in warp
//! @param  left   lower limit of interval
//! @param  right  upper limit of interval
//! @param  left_count  eigenvalues less than \a left
//! @param  right_count  eigenvalues less than \a right
//! @param  all_threads_converged  shared memory flag if all threads are
//!                                 converged
///////////////////////////////////////////////////////////////////////////////

 
template<class T>
void generate_subdivideActiveInterval(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     subdivideActiveInterval(const unsigned int tid,  \n");
    source.append("                             float *s_left, float *s_right,  \n");
    source.append("                             __global "); source.append(numeric_string); source.append("*s_left_count, T *s_right_count,  \n");
    source.append("                             const unsigned int num_threads_active,  \n");
    source.append("                             float &left, float &right,  \n");
    source.append("                             unsigned int &left_count, unsigned int &right_count,  \n");
    source.append("                             float &mid, unsigned int &all_threads_converged)  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");

        // for all active threads
    source.append("         if (tid < num_threads_active)  \n");
    source.append("         {  \n");

    source.append("             left = s_left[tid];  \n");
    source.append("             right = s_right[tid];  \n");
    source.append("             left_count = s_left_count[tid];  \n");
    source.append("             right_count = s_right_count[tid];  \n");

            // check if thread already converged
            //if( std::abs(left - right) > 0.00000001f )
    source.append("             if (left != right)  \n");
    source.append("             {  \n");

    source.append("                 mid = computeMidpoint(left, right);  \n");
    source.append("                 all_threads_converged = 0;  \n");
    source.append("             }  \n");
    source.append("             else if ((right_count - left_count) > 1)  \n");
    source.append("             {  \n");
                // mark as not converged if multiple eigenvalues enclosed
                // duplicate interval in storeIntervalsConverged()
    source.append("                 all_threads_converged = 0;  \n");
    source.append("             }  \n");

    source.append("         }    \n");
// end for all active threads
    source.append("     }  \n");
} 
  //}
//}

#endif // #ifndef _BISECT_UTIL_H_





 
