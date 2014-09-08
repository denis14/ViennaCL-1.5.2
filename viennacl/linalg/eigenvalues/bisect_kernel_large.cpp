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

//namespace viennacl
//{
  //namespace linalg
  //{

    ////////////////////////////////////////////////////////////////////////////////
    //! Write data to global memory
    ////////////////////////////////////////////////////////////////////////////////

void generate_writeToGmem(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void writeToGmem(const unsigned int tid, const unsigned int tid_2,  \n");
    source.append("                      const unsigned int num_threads_active,  \n");
    source.append("                      const unsigned int num_blocks_mult,  \n");
    source.append("                      float *g_left_one, float *g_right_one,  \n");
    source.append("                      unsigned int *g_pos_one,  \n");
    source.append("                      float *g_left_mult, float *g_right_mult,  \n");
    source.append("                      unsigned int *g_left_count_mult,  \n");
    source.append("                      unsigned int *g_right_count_mult,  \n");
    source.append("                      float *s_left, float *s_right,  \n");
    source.append("                      unsigned short *s_left_count, unsigned short *s_right_count,  \n");
    source.append("                      unsigned int *g_blocks_mult,  \n");
    source.append("                      unsigned int *g_blocks_mult_sum,  \n");
    source.append("                      unsigned short *s_compaction_list,  \n");
    source.append("                      unsigned short *s_cl_helper,  \n");
    source.append("                      unsigned int offset_mult_lambda  \n");
    source.append("                     );  \n");

} 



////////////////////////////////////////////////////////////////////////////////
//! Perform final stream compaction before writing out data
////////////////////////////////////////////////////////////////////////////////

void generate_compactStreamsFinal(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     compactStreamsFinal(const unsigned int tid, const unsigned int tid_2,  \n");
    source.append("                         const unsigned int num_threads_active,  \n");
    source.append("                         unsigned int &offset_mult_lambda,  \n");
    source.append("                         float *s_left, float *s_right,  \n");
    source.append("                         unsigned short *s_left_count, unsigned short *s_right_count,  \n");
    source.append("                         unsigned short *s_cl_one, unsigned short *s_cl_mult,  \n");
    source.append("                         unsigned short *s_cl_blocking, unsigned short *s_cl_helper,  \n");
    source.append("                         unsigned int is_one_lambda, unsigned int is_one_lambda_2,  \n");
    source.append("                         float &left, float &right, float &left_2, float &right_2,  \n");
    source.append("                         unsigned int &left_count, unsigned int &right_count,  \n");
    source.append("                         unsigned int &left_count_2, unsigned int &right_count_2,  \n");
    source.append("                         unsigned int c_block_iend, unsigned int c_sum_block,  \n");
    source.append("                         unsigned int c_block_iend_2, unsigned int c_sum_block_2  \n");
    source.append("                        );  \n");


} 


////////////////////////////////////////////////////////////////////////////////
//! Perform scan to compact list of block start addresses
////////////////////////////////////////////////////////////////////////////////

void generate_scanCompactBlocksStartAddress(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     scanCompactBlocksStartAddress(const unsigned int tid, const unsigned int tid_2,  \n");
    source.append("                                   const unsigned int num_threads_compaction,  \n");
    source.append("                                   unsigned short *s_cl_blocking,  \n");
    source.append("                                   unsigned short *s_cl_helper  \n");
    source.append("                                  );  \n");


} 


////////////////////////////////////////////////////////////////////////////////
//! Perform scan to obtain number of eigenvalues before a specific block
////////////////////////////////////////////////////////////////////////////////

void generate_scanSumBlocks(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     scanSumBlocks(const unsigned int tid, const unsigned int tid_2,  \n");
    source.append("                   const unsigned int num_threads_active,  \n");
    source.append("                   const unsigned int num_threads_compaction,  \n");
    source.append("                   unsigned short *s_cl_blocking,  \n");
    source.append("                   unsigned short *s_cl_helper  \n");
    source.append("                  );  \n");


} 


////////////////////////////////////////////////////////////////////////////////
//! Perform initial scan for compaction of intervals containing one and
//! multiple eigenvalues; also do initial scan to build blocks
////////////////////////////////////////////////////////////////////////////////

void generate_scanInitial(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     scanInitial(const unsigned int tid, const unsigned int tid_2,  \n");
    source.append("                 const unsigned int num_threads_active,  \n");
    source.append("                 const unsigned int num_threads_compaction,  \n");
    source.append("                 unsigned short *s_cl_one, unsigned short *s_cl_mult,  \n");
    source.append("                 unsigned short *s_cl_blocking, unsigned short *s_cl_helper  \n");
    source.append("                );  \n");

} 



////////////////////////////////////////////////////////////////////////////////
//! Store all non-empty intervals resulting from the subdivision of the interval
//! currently processed by the thread
//! @param  addr  address where to store
////////////////////////////////////////////////////////////////////////////////

void generate_storeNonEmptyIntervalsLarge(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     storeNonEmptyIntervalsLarge(unsigned int addr,  \n");
    source.append("                                 const unsigned int num_threads_active,  \n");
    source.append("                                 float  *s_left, float *s_right,  \n");
    source.append("                                 unsigned short  *s_left_count,  \n");
    source.append("                                 unsigned short *s_right_count,  \n");
    source.append("                                 float left, float mid, float right,  \n");
    source.append("                                 const unsigned short left_count,  \n");
    source.append("                                 const unsigned short mid_count,  \n");
    source.append("                                 const unsigned short right_count,  \n");
    source.append("                                 float epsilon,  \n");
    source.append("                                 unsigned int &compact_second_chunk,  \n");
    source.append("                                 unsigned short *s_compaction_list,  \n");
    source.append("                                 unsigned int &is_active_second  \n");
    source.append("                                );  \n");


} 

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
 

void generate_bisectKernelLarge(StringType & source, std::string const & numeric_string)
{
    source.append("     __kernel  \n");
    source.append("     void  \n");
    source.append("     bisectKernelLarge(float *g_d, float *g_s, const unsigned int n,  \n");
    source.append("                       const float lg, const float ug,  \n");
    source.append("                       const unsigned int lg_eig_count,  \n");
    source.append("                       const unsigned int ug_eig_count,  \n");
    source.append("                       float epsilon,  \n");
    source.append("                       unsigned int *g_num_one,  \n");
    source.append("                       unsigned int *g_num_blocks_mult,  \n");
    source.append("                       float *g_left_one, float *g_right_one,  \n");
    source.append("                       unsigned int *g_pos_one,  \n");
    source.append("                       float *g_left_mult, float *g_right_mult,  \n");
    source.append("                       unsigned int *g_left_count_mult,  \n");
    source.append("                       unsigned int *g_right_count_mult,  \n");
    source.append("                       unsigned int *g_blocks_mult,  \n");
    source.append("                       unsigned int *g_blocks_mult_sum  \n");
    source.append("                      )  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");

    source.append("         const unsigned int tid = lcl_id;  \n");

        // intervals (store left and right because the subdivision tree is in general
        // not dense
    source.append("         __local  float  s_left[2 * MAX_THREADS_BLOCK + 1];  \n");
    source.append("         __local  float  s_right[2 * MAX_THREADS_BLOCK + 1];  \n");

        // number of eigenvalues that are smaller than s_left / s_right
        // (correspondence is realized via indices)
    source.append("         __local  unsigned short  s_left_count[2 * MAX_THREADS_BLOCK + 1];  \n");
    source.append("         __local  unsigned short  s_right_count[2 * MAX_THREADS_BLOCK + 1];  \n");

        // helper for stream compaction
    source.append("         __local  unsigned short  s_compaction_list[2 * MAX_THREADS_BLOCK + 1];  \n");

        // state variables for whole block
        // if 0 then compaction of second chunk of child intervals is not necessary
        // (because all intervals had exactly one non-dead child)
    source.append("         __local  unsigned int compact_second_chunk;  \n");
        // if 1 then all threads are converged
    source.append("         __local  unsigned int all_threads_converged;  \n");

        // number of currently active threads
    source.append("         __local  unsigned int num_threads_active;  \n");

        // number of threads to use for stream compaction
    source.append("         __local  unsigned int num_threads_compaction;  \n");

        // helper for exclusive scan
    source.append("         unsigned short *s_compaction_list_exc = s_compaction_list + 1;  \n");


        // variables for currently processed interval
        // left and right limit of active interval
    source.append("         float left = 0.0f;  \n");
    source.append("         float right = 0.0f;  \n");
    source.append("         unsigned int left_count = 0;  \n");
    source.append("         unsigned int right_count = 0;  \n");
        // midpoint of active interval
    source.append("         float  mid = 0.0f;  \n");
        // number of eigenvalues smaller then mid
    source.append("         unsigned int mid_count = 0;  \n");
        // helper for stream compaction (tracking of threads generating second child)
    source.append("         unsigned int is_active_second = 0;  \n");

        // initialize lists
    source.append("         s_compaction_list[tid] = 0;  \n");
    source.append("         s_left[tid] = 0;  \n");
    source.append("         s_right[tid] = 0;  \n");
    source.append("         s_left_count[tid] = 0;  \n");
    source.append("         s_right_count[tid] = 0;  \n");

    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

        // set up initial configuration
    source.append("         if (0 == tid)  \n");
    source.append("         {  \n");

    source.append("             s_left[0] = lg;  \n");
    source.append("             s_right[0] = ug;  \n");
    source.append("             s_left_count[0] = lg_eig_count;  \n");
    source.append("             s_right_count[0] = ug_eig_count;  \n");

    source.append("             compact_second_chunk = 0;  \n");
    source.append("             num_threads_active = 1;  \n");

    source.append("             num_threads_compaction = 1;  \n");

    source.append("             all_threads_converged = 1;  \n");
    source.append("         }  \n");

    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

        // for all active threads read intervals from the last level
        // the number of (worst case) active threads per level l is 2^l
        //while (true)                                                       
    source.append("    for( unsigned int i = 0; i < 15; ++i )    \n");                             // selbst hinzugefuegt
    source.append("         {  \n");
    source.append("             s_compaction_list[tid] = 0;  \n");
    source.append("             s_compaction_list[tid + MAX_THREADS_BLOCK] = 0;  \n");
    source.append("             s_compaction_list[2 * MAX_THREADS_BLOCK] = 0;  \n");
    source.append("             subdivideActiveInterval(tid, s_left, s_right, s_left_count, s_right_count,  \n");
    source.append("                                     num_threads_active,  \n");
    source.append("                                     left, right, left_count, right_count,  \n");
    source.append("                                     mid, all_threads_converged);  \n");

    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

            // check if done
    source.append("             if (1 == all_threads_converged)  \n");
    source.append("             {  \n");
    source.append("                 break;  \n");
    source.append("             }  \n");

            // compute number of eigenvalues smaller than mid
            // use all threads for reading the necessary matrix data from global
            // memory
            // use s_left and s_right as scratch space for diagonal and
            // superdiagonal of matrix
    source.append("             mid_count = computeNumSmallerEigenvalsLarge(g_d, g_s, n,  \n");
    source.append("                                                         mid, lcl_id,  \n");
    source.append("                                                         num_threads_active,  \n");
    source.append("                                                         s_left, s_right,  \n");
    source.append("                                                         (left == right));  \n");

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
    source.append("             if (tid < num_threads_active)  \n");
    source.append("             {  \n");

    source.append("                 if (left != right)  \n");
    source.append("                 {  \n");

                    // store intervals
    source.append("                     storeNonEmptyIntervalsLarge(tid, num_threads_active,  \n");
    source.append("                                                 s_left, s_right,  \n");
    source.append("                                                 s_left_count, s_right_count,  \n");
    source.append("                                                 left, mid, right,  \n");
    source.append("                                                 left_count, mid_count, right_count,  \n");
    source.append("                                                 epsilon, compact_second_chunk,  \n");
    source.append("                                                 s_compaction_list_exc,  \n");
    source.append("                                                 is_active_second);  \n");
    source.append("                 }  \n");
    source.append("                 else  \n");
    source.append("                 {  \n");

                    // re-write converged interval (has to be stored again because s_left
                    // and s_right are used as scratch space for
                    // computeNumSmallerEigenvalsLarge()
    source.append("                     s_left[tid] = left;  \n");
    source.append("                     s_right[tid] = left;  \n");
    source.append("                     s_left_count[tid] = left_count;  \n");
    source.append("                     s_right_count[tid] = right_count;  \n");

    source.append("                     is_active_second = 0;  \n");
    source.append("                 }  \n");
    source.append("             }  \n");

            // necessary so that compact_second_chunk is up-to-date
    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

            // perform compaction of chunk where second children are stored
            // scan of (num_threads_active / 2) elements, thus at most
            // (num_threads_active / 4) threads are needed
    source.append("             if (compact_second_chunk > 0)  \n");
    source.append("             {  \n");

                // create indices for compaction
    source.append("                 createIndicesCompaction(s_compaction_list_exc, num_threads_compaction);  \n");
    source.append("             }  \n");
    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
    source.append("               \n");
    source.append("        if (compact_second_chunk > 0)               \n");                               // selbst veraendert
    source.append("             {  \n");
    source.append("                 compactIntervals(s_left, s_right, s_left_count, s_right_count,  \n");
    source.append("                                  mid, right, mid_count, right_count,  \n");
    source.append("                                  s_compaction_list, num_threads_active,  \n");
    source.append("                                  is_active_second);  \n");
    source.append("             }  \n");

    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

            // update state variables
    source.append("             if (0 == tid)  \n");
    source.append("             {  \n");

                // update number of active threads with result of reduction
    source.append("                 num_threads_active += s_compaction_list[num_threads_active];  \n");
    source.append("                 num_threads_compaction = ceilPow2(num_threads_active);  \n");

    source.append("                 compact_second_chunk = 0;  \n");
    source.append("                 all_threads_converged = 1;  \n");
    source.append("             }  \n");

    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

    source.append("             if (num_threads_compaction > lcl_sz)  \n");
    source.append("             {  \n");
    source.append("                 break;  \n");
    source.append("             }  \n");

    source.append("         }  \n");

    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

        // generate two lists of intervals; one with intervals that contain one
        // eigenvalue (or are converged), and one with intervals that need further
        // subdivision

        // perform two scans in parallel

    source.append("         unsigned int left_count_2;  \n");
    source.append("         unsigned int right_count_2;  \n");

    source.append("         unsigned int tid_2 = tid + lcl_sz;  \n");

        // cache in per thread registers so that s_left_count and s_right_count
        // can be used for scans
    source.append("         left_count = s_left_count[tid];  \n");
    source.append("         right_count = s_right_count[tid];  \n");

        // some threads have to cache data for two intervals
    source.append("         if (tid_2 < num_threads_active)  \n");
    source.append("         {  \n");
    source.append("             left_count_2 = s_left_count[tid_2];  \n");
    source.append("             right_count_2 = s_right_count[tid_2];  \n");
    source.append("         }  \n");

        // compaction list for intervals containing one and multiple eigenvalues
        // do not affect first element for exclusive scan
    source.append("         unsigned short  *s_cl_one = s_left_count + 1;  \n");
    source.append("         unsigned short  *s_cl_mult = s_right_count + 1;  \n");

        // compaction list for generating blocks of intervals containing multiple
        // eigenvalues
    source.append("         unsigned short  *s_cl_blocking = s_compaction_list_exc;  \n");
        // helper compaction list for generating blocks of intervals
    source.append("         __local unsigned short  s_cl_helper[2 * MAX_THREADS_BLOCK + 1];  \n");

    source.append("         if (0 == tid)  \n");
    source.append("         {  \n");
            // set to 0 for exclusive scan
    source.append("             s_left_count[0] = 0;  \n");
    source.append("             s_right_count[0] = 0;  \n");
    source.append("              \n");
    source.append("         }  \n");

    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

        // flag if interval contains one or multiple eigenvalues
    source.append("         unsigned int is_one_lambda = 0;  \n");
    source.append("         unsigned int is_one_lambda_2 = 0;  \n");

        // number of eigenvalues in the interval
    source.append("         unsigned int multiplicity = right_count - left_count;  \n");
    source.append("         is_one_lambda = (1 == multiplicity);  \n");

    source.append("         s_cl_one[tid] = is_one_lambda;  \n");
    source.append("         s_cl_mult[tid] = (! is_one_lambda);  \n");

        // (note: s_cl_blocking is non-zero only where s_cl_mult[] is non-zero)
    source.append("         s_cl_blocking[tid] = (1 == is_one_lambda) ? 0 : multiplicity;  \n");
    source.append("         s_cl_helper[tid] = 0;  \n");

    source.append("         if (tid_2 < num_threads_active)  \n");
    source.append("         {  \n");

    source.append("             unsigned int multiplicity = right_count_2 - left_count_2;  \n");
    source.append("             is_one_lambda_2 = (1 == multiplicity);  \n");

    source.append("             s_cl_one[tid_2] = is_one_lambda_2;  \n");
    source.append("             s_cl_mult[tid_2] = (! is_one_lambda_2);  \n");

            // (note: s_cl_blocking is non-zero only where s_cl_mult[] is non-zero)
    source.append("             s_cl_blocking[tid_2] = (1 == is_one_lambda_2) ? 0 : multiplicity;  \n");
    source.append("             s_cl_helper[tid_2] = 0;  \n");
    source.append("         }  \n");
    source.append("         else if (tid_2 < (2 * MAX_THREADS_BLOCK + 1))  \n");
    source.append("         {  \n");

            // clear
    source.append("             s_cl_blocking[tid_2] = 0;  \n");
    source.append("             s_cl_helper[tid_2] = 0;  \n");
    source.append("         }  \n");


    source.append("         scanInitial(tid, tid_2, num_threads_active, num_threads_compaction,  \n");
    source.append("                     s_cl_one, s_cl_mult, s_cl_blocking, s_cl_helper);  \n");
    source.append("           \n");
    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");                                                 // selbst hinzugefuegt

    source.append("         scanSumBlocks(tid, tid_2, num_threads_active,  \n");
    source.append("                       num_threads_compaction, s_cl_blocking, s_cl_helper);  \n");

        // end down sweep of scan
    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

    source.append("         unsigned int  c_block_iend = 0;  \n");
    source.append("         unsigned int  c_block_iend_2 = 0;  \n");
    source.append("         unsigned int  c_sum_block = 0;  \n");
    source.append("         unsigned int  c_sum_block_2 = 0;  \n");

        // for each thread / interval that corresponds to root node of interval block
        // store start address of block and total number of eigenvalues in all blocks
        // before this block (particular thread is irrelevant, constraint is to
        // have a subset of threads so that one and only one of them is in each
        // interval)
    source.append("         if (1 == s_cl_helper[tid])  \n");
    source.append("         {  \n");

    source.append("             c_block_iend = s_cl_mult[tid] + 1;  \n");
    source.append("             c_sum_block = s_cl_blocking[tid];  \n");
    source.append("         }  \n");

    source.append("         if (1 == s_cl_helper[tid_2])  \n");
    source.append("         {  \n");

    source.append("             c_block_iend_2 = s_cl_mult[tid_2] + 1;  \n");
    source.append("             c_sum_block_2 = s_cl_blocking[tid_2];  \n");
    source.append("         }  \n");

    source.append("         scanCompactBlocksStartAddress(tid, tid_2, num_threads_compaction,  \n");
    source.append("                                       s_cl_blocking, s_cl_helper);  \n");


        // finished second scan for s_cl_blocking
    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

        // determine the global results
    source.append("         __local  unsigned int num_blocks_mult;  \n");
    source.append("         __local  unsigned int num_mult;  \n");
    source.append("         __local  unsigned int offset_mult_lambda;  \n");

    source.append("         if (0 == tid)  \n");
    source.append("         {  \n");

    source.append("             num_blocks_mult = s_cl_blocking[num_threads_active - 1];  \n");
    source.append("             offset_mult_lambda = s_cl_one[num_threads_active - 1];  \n");
    source.append("             num_mult = s_cl_mult[num_threads_active - 1];  \n");

    source.append("             *g_num_one = offset_mult_lambda;  \n");
    source.append("             *g_num_blocks_mult = num_blocks_mult;  \n");
    source.append("         }  \n");

    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

    source.append("         float left_2, right_2;  \n");
    source.append("         --s_cl_one;  \n");
    source.append("         --s_cl_mult;  \n");
    source.append("         --s_cl_blocking;  \n");
    source.append("           \n");
    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");                                               // selbst hinzugefuegt
    source.append("         compactStreamsFinal(tid, tid_2, num_threads_active, offset_mult_lambda,  \n");
    source.append("                             s_left, s_right, s_left_count, s_right_count,  \n");
    source.append("                             s_cl_one, s_cl_mult, s_cl_blocking, s_cl_helper,  \n");
    source.append("                             is_one_lambda, is_one_lambda_2,  \n");
    source.append("                             left, right, left_2, right_2,  \n");
    source.append("                             left_count, right_count, left_count_2, right_count_2,  \n");
    source.append("                             c_block_iend, c_sum_block, c_block_iend_2, c_sum_block_2  \n");
    source.append("                            );  \n");

    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

        // final adjustment before writing out data to global memory
    source.append("         if (0 == tid)  \n");
    source.append("         {  \n");
    source.append("             s_cl_blocking[num_blocks_mult] = num_mult;  \n");
    source.append("             s_cl_helper[0] = 0;  \n");
    source.append("         }  \n");

    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

        // write to global memory
    source.append("         writeToGmem(tid, tid_2, num_threads_active, num_blocks_mult,  \n");
    source.append("                     g_left_one, g_right_one, g_pos_one,  \n");
    source.append("                     g_left_mult, g_right_mult, g_left_count_mult, g_right_count_mult,  \n");
    source.append("                     s_left, s_right, s_left_count, s_right_count,  \n");
    source.append("                     g_blocks_mult, g_blocks_mult_sum,  \n");
    source.append("                     s_compaction_list, s_cl_helper, offset_mult_lambda);  \n");
    source.append("                       \n");
    source.append("     }  \n");


} 


////////////////////////////////////////////////////////////////////////////////
//! Write data to global memory
////////////////////////////////////////////////////////////////////////////////

void generate_writeToGmem(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void writeToGmem(const unsigned int tid, const unsigned int tid_2,  \n");
    source.append("                      const unsigned int num_threads_active,  \n");
    source.append("                      const unsigned int num_blocks_mult,  \n");
    source.append("                      float *g_left_one, float *g_right_one,  \n");
    source.append("                      unsigned int *g_pos_one,  \n");
    source.append("                      float *g_left_mult, float *g_right_mult,  \n");
    source.append("                      unsigned int *g_left_count_mult,  \n");
    source.append("                      unsigned int *g_right_count_mult,  \n");
    source.append("                      float *s_left, float *s_right,  \n");
    source.append("                      unsigned short *s_left_count, unsigned short *s_right_count,  \n");
    source.append("                      unsigned int *g_blocks_mult,  \n");
    source.append("                      unsigned int *g_blocks_mult_sum,  \n");
    source.append("                      unsigned short *s_compaction_list,  \n");
    source.append("                      unsigned short *s_cl_helper,  \n");
    source.append("                      unsigned int offset_mult_lambda  \n");
    source.append("                     )  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");


    source.append("         if (tid < offset_mult_lambda)  \n");
    source.append("         {  \n");

    source.append("             g_left_one[tid] = s_left[tid];  \n");
    source.append("             g_right_one[tid] = s_right[tid];  \n");
            // right count can be used to order eigenvalues without sorting
    source.append("             g_pos_one[tid] = s_right_count[tid];  \n");
    source.append("         }  \n");
    source.append("         else  \n");
    source.append("         {  \n");

    source.append("               \n");
    source.append("             g_left_mult[tid - offset_mult_lambda] = s_left[tid];  \n");
    source.append("             g_right_mult[tid - offset_mult_lambda] = s_right[tid];  \n");
    source.append("             g_left_count_mult[tid - offset_mult_lambda] = s_left_count[tid];  \n");
    source.append("             g_right_count_mult[tid - offset_mult_lambda] = s_right_count[tid];  \n");
    source.append("         }  \n");

    source.append("         if (tid_2 < num_threads_active)  \n");
    source.append("         {  \n");

    source.append("             if (tid_2 < offset_mult_lambda)  \n");
    source.append("             {  \n");

    source.append("                 g_left_one[tid_2] = s_left[tid_2];  \n");
    source.append("                 g_right_one[tid_2] = s_right[tid_2];  \n");
                // right count can be used to order eigenvalues without sorting
    source.append("                 g_pos_one[tid_2] = s_right_count[tid_2];  \n");
    source.append("             }  \n");
    source.append("             else  \n");
    source.append("             {  \n");

    source.append("                 g_left_mult[tid_2 - offset_mult_lambda] = s_left[tid_2];  \n");
    source.append("                 g_right_mult[tid_2 - offset_mult_lambda] = s_right[tid_2];  \n");
    source.append("                 g_left_count_mult[tid_2 - offset_mult_lambda] = s_left_count[tid_2];  \n");
    source.append("                 g_right_count_mult[tid_2 - offset_mult_lambda] = s_right_count[tid_2];  \n");
    source.append("             }  \n");

    source.append("    } \n");      // end writing out data

        source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");                                    // selbst hinzugefuegt

        // note that s_cl_blocking = s_compaction_list + 1;, that is by writing out
        // s_compaction_list we write the exclusive scan result
    source.append("         if (tid <= num_blocks_mult)  \n");
    source.append("         {  \n");
    source.append("             g_blocks_mult[tid] = s_compaction_list[tid];  \n");
    source.append("             g_blocks_mult_sum[tid] = s_cl_helper[tid];  \n");
    source.append("         }  \n");

    source.append("         if (tid_2 <= num_blocks_mult)  \n");
    source.append("         {  \n");
    source.append("             g_blocks_mult[tid_2] = s_compaction_list[tid_2];  \n");
    source.append("             g_blocks_mult_sum[tid_2] = s_cl_helper[tid_2];  \n");
    source.append("         }  \n");
    source.append("     }  \n");


} 


////////////////////////////////////////////////////////////////////////////////
//! Perform final stream compaction before writing data to global memory
////////////////////////////////////////////////////////////////////////////////

void generate_compactStreamsFinal(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     compactStreamsFinal(const unsigned int tid, const unsigned int tid_2,  \n");
    source.append("                         const unsigned int num_threads_active,  \n");
    source.append("                         unsigned int &offset_mult_lambda,  \n");
    source.append("                         float *s_left, float *s_right,  \n");
    source.append("                         unsigned short *s_left_count, unsigned short *s_right_count,  \n");
    source.append("                         unsigned short *s_cl_one, unsigned short *s_cl_mult,  \n");
    source.append("                         unsigned short *s_cl_blocking, unsigned short *s_cl_helper,  \n");
    source.append("                         unsigned int is_one_lambda, unsigned int is_one_lambda_2,  \n");
    source.append("                         float &left, float &right, float &left_2, float &right_2,  \n");
    source.append("                         unsigned int &left_count, unsigned int &right_count,  \n");
    source.append("                         unsigned int &left_count_2, unsigned int &right_count_2,  \n");
    source.append("                         unsigned int c_block_iend, unsigned int c_sum_block,  \n");
    source.append("                         unsigned int c_block_iend_2, unsigned int c_sum_block_2  \n");
    source.append("                        )  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");

        // cache data before performing compaction
    source.append("         left = s_left[tid];  \n");
    source.append("         right = s_right[tid];  \n");

    source.append("         if (tid_2 < num_threads_active)  \n");
    source.append("         {  \n");

    source.append("             left_2 = s_left[tid_2];  \n");
    source.append("             right_2 = s_right[tid_2];  \n");
    source.append("         }  \n");

    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

        // determine addresses for intervals containing multiple eigenvalues and
        // addresses for blocks of intervals
    source.append("         unsigned int ptr_w = 0;  \n");
    source.append("         unsigned int ptr_w_2 = 0;  \n");
    source.append("         unsigned int ptr_blocking_w = 0;  \n");
    source.append("         unsigned int ptr_blocking_w_2 = 0;  \n");
    source.append("           \n");
    source.append("          \n");

    source.append("         ptr_w = (1 == is_one_lambda) ? s_cl_one[tid]  \n");
    source.append("                 : s_cl_mult[tid] + offset_mult_lambda;  \n");

    source.append("         if (0 != c_block_iend)  \n");
    source.append("         {  \n");
    source.append("             ptr_blocking_w = s_cl_blocking[tid];  \n");
    source.append("         }  \n");

    source.append("         if (tid_2 < num_threads_active)  \n");
    source.append("         {  \n");
    source.append("             ptr_w_2 = (1 == is_one_lambda_2) ? s_cl_one[tid_2]  \n");
    source.append("                       : s_cl_mult[tid_2] + offset_mult_lambda;  \n");

    source.append("             if (0 != c_block_iend_2)  \n");
    source.append("             {  \n");
    source.append("                 ptr_blocking_w_2 = s_cl_blocking[tid_2];  \n");
    source.append("             }  \n");
    source.append("         }  \n");
    source.append("           \n");
    source.append("          \n");
    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
    source.append("           \n");
        // store compactly in shared mem
    source.append("           s_left[ptr_w] = left;  \n");
    source.append("           s_right[ptr_w] = right;  \n");
    source.append("           s_left_count[ptr_w] = left_count;  \n");
    source.append("           s_right_count[ptr_w] = right_count;  \n");
    source.append("           \n");
    source.append("           \n");

    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");                                                     // selbst hinzugefuegt
    source.append("         if(tid == 1)  \n");
    source.append("         {  \n");
    source.append("           s_left[ptr_w] = left;  \n");
    source.append("           s_right[ptr_w] = right;  \n");
    source.append("           s_left_count[ptr_w] = left_count;  \n");
    source.append("           s_right_count[ptr_w] = right_count;  \n");
          //printf("s_l = %10.8f\t left = %10.8f\t s_r = %10.8f\t s_l_c = %u\t s_r_c = %u\t tid = %u \t ptr_w = %u\n",
            //s_left[ptr_w], left, s_right[ptr_w], s_left_count[ptr_w], s_right_count[ptr_w], tid, ptr_w);

    source.append("           \n");
    source.append("         }  \n");
    source.append("               if (0 != c_block_iend)  \n");
    source.append("         {  \n");
    source.append("             s_cl_blocking[ptr_blocking_w + 1] = c_block_iend - 1;  \n");
    source.append("             s_cl_helper[ptr_blocking_w + 1] = c_sum_block;  \n");
    source.append("         }  \n");
    source.append("           \n");
    source.append("         if (tid_2 < num_threads_active)  \n");
    source.append("         {  \n");

            // store compactly in shared mem
    source.append("             s_left[ptr_w_2] = left_2;  \n");
    source.append("             s_right[ptr_w_2] = right_2;  \n");
    source.append("             s_left_count[ptr_w_2] = left_count_2;  \n");
    source.append("             s_right_count[ptr_w_2] = right_count_2;  \n");

    source.append("             if (0 != c_block_iend_2)  \n");
    source.append("             {  \n");
    source.append("                 s_cl_blocking[ptr_blocking_w_2 + 1] = c_block_iend_2 - 1;  \n");
    source.append("                 s_cl_helper[ptr_blocking_w_2 + 1] = c_sum_block_2;  \n");
    source.append("             }  \n");
    source.append("         }  \n");

    source.append("     }  \n");


} 



////////////////////////////////////////////////////////////////////////////////
//! Compute addresses to obtain compact list of block start addresses
////////////////////////////////////////////////////////////////////////////////

void generate_scanCompactBlocksStartAddress(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     scanCompactBlocksStartAddress(const unsigned int tid, const unsigned int tid_2,  \n");
    source.append("                                   const unsigned int num_threads_compaction,  \n");
    source.append("                                   unsigned short *s_cl_blocking,  \n");
    source.append("                                   unsigned short *s_cl_helper  \n");
    source.append("                                  )  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");

        // prepare for second step of block generation: compaction of the block
        // list itself to efficiently write out these
    source.append("         s_cl_blocking[tid] = s_cl_helper[tid];  \n");

    source.append("         if (tid_2 < num_threads_compaction)  \n");
    source.append("         {  \n");
    source.append("             s_cl_blocking[tid_2] = s_cl_helper[tid_2];  \n");
    source.append("         }  \n");

    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

        // additional scan to compact s_cl_blocking that permits to generate a
        // compact list of eigenvalue blocks each one containing about
        // MAX_THREADS_BLOCK eigenvalues (so that each of these blocks may be
        // processed by one thread block in a subsequent processing step

    source.append("         unsigned int offset = 1;  \n");

        // build scan tree
    source.append("         for (int d = (num_threads_compaction >> 1); d > 0; d >>= 1)  \n");
    source.append("         {  \n");

    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

    source.append("             if (tid < d)  \n");
    source.append("             {  \n");

    source.append("                 unsigned int  ai = offset*(2*tid+1)-1;  \n");
    source.append("                 unsigned int  bi = offset*(2*tid+2)-1;  \n");
    source.append("                 s_cl_blocking[bi] = s_cl_blocking[bi] + s_cl_blocking[ai];  \n");
    source.append("             }  \n");

    source.append("             offset <<= 1;  \n");
    source.append("         }  \n");

        // traverse down tree: first down to level 2 across
    source.append("         for (int d = 2; d < num_threads_compaction; d <<= 1)  \n");
    source.append("         {  \n");

    source.append("             offset >>= 1;  \n");
    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

            //
    source.append("             if (tid < (d-1))  \n");
    source.append("             {  \n");

    source.append("                 unsigned int  ai = offset*(tid+1) - 1;  \n");
    source.append("                 unsigned int  bi = ai + (offset >> 1);  \n");
    source.append("                 s_cl_blocking[bi] = s_cl_blocking[bi] + s_cl_blocking[ai];  \n");
    source.append("             }  \n");
    source.append("         }  \n");

    source.append("     }  \n");


} 


////////////////////////////////////////////////////////////////////////////////
//! Perform scan to obtain number of eigenvalues before a specific block
////////////////////////////////////////////////////////////////////////////////

void generate_scanSumBlocks(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     scanSumBlocks(const unsigned int tid, const unsigned int tid_2,  \n");
    source.append("                   const unsigned int num_threads_active,  \n");
    source.append("                   const unsigned int num_threads_compaction,  \n");
    source.append("                   unsigned short *s_cl_blocking,  \n");
    source.append("                   unsigned short *s_cl_helper)  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");

    source.append("         unsigned int offset = 1;  \n");

        // first step of scan to build the sum of elements within each block
        // build up tree
    source.append("         for (int d = num_threads_compaction >> 1; d > 0; d >>= 1)  \n");
    source.append("         {  \n");

    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

    source.append("             if (tid < d)  \n");
    source.append("             {  \n");

    source.append("                 unsigned int ai = offset*(2*tid+1)-1;  \n");
    source.append("                 unsigned int bi = offset*(2*tid+2)-1;  \n");

    source.append("                 s_cl_blocking[bi] += s_cl_blocking[ai];  \n");
    source.append("             }  \n");

    source.append("             offset *= 2;  \n");
    source.append("         }  \n");

        // first step of scan to build the sum of elements within each block
        // traverse down tree
    source.append("         for (int d = 2; d < (num_threads_compaction - 1); d <<= 1)  \n");
    source.append("         {  \n");

    source.append("             offset >>= 1;  \n");
    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

    source.append("             if (tid < (d-1))  \n");
    source.append("             {  \n");

    source.append("                 unsigned int ai = offset*(tid+1) - 1;  \n");
    source.append("                 unsigned int bi = ai + (offset >> 1);  \n");

    source.append("                 s_cl_blocking[bi] += s_cl_blocking[ai];  \n");
    source.append("             }  \n");
    source.append("         }  \n");

    source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

    source.append("         if (0 == tid)  \n");
    source.append("         {  \n");

            // move last element of scan to last element that is valid
            // necessary because the number of threads employed for scan is a power
            // of two and not necessarily the number of active threasd
    source.append("             s_cl_helper[num_threads_active - 1] =  \n");
    source.append("                 s_cl_helper[num_threads_compaction - 1];  \n");
    source.append("             s_cl_blocking[num_threads_active - 1] =  \n");
    source.append("                 s_cl_blocking[num_threads_compaction - 1];  \n");
    source.append("         }  \n");
    source.append("     }  \n");


} 

////////////////////////////////////////////////////////////////////////////////
//! Perform initial scan for compaction of intervals containing one and
//! multiple eigenvalues; also do initial scan to build blocks
////////////////////////////////////////////////////////////////////////////////
 

void generate_scanInitial(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     scanInitial(const unsigned int tid, const unsigned int tid_2,  \n");
    source.append("                 const unsigned int num_threads_active,  \n");
    source.append("                 const unsigned int num_threads_compaction,  \n");
    source.append("                 unsigned short *s_cl_one, unsigned short *s_cl_mult,  \n");
    source.append("                 unsigned short *s_cl_blocking, unsigned short *s_cl_helper  \n");
    source.append("                )  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");


        // perform scan to compactly write out the intervals containing one and
        // multiple eigenvalues
        // also generate tree for blocking of intervals containing multiple
        // eigenvalues

    source.append("         unsigned int offset = 1;  \n");

        // build scan tree
    source.append("         for (int d = (num_threads_compaction >> 1); d > 0; d >>= 1)  \n");
    source.append("         {  \n");

    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

    source.append("             if (tid < d)  \n");
    source.append("             {  \n");

    source.append("                 unsigned int  ai = offset*(2*tid+1);  \n");
    source.append("                 unsigned int  bi = offset*(2*tid+2)-1;  \n");

    source.append("                 s_cl_one[bi] = s_cl_one[bi] + s_cl_one[ai - 1];  \n");
    source.append("                 s_cl_mult[bi] = s_cl_mult[bi] + s_cl_mult[ai - 1];  \n");

                // s_cl_helper is binary and zero for an internal node and 1 for a
                // root node of a tree corresponding to a block
                // s_cl_blocking contains the number of nodes in each sub-tree at each
                // iteration, the data has to be kept to compute the total number of
                // eigenvalues per block that, in turn, is needed to efficiently
                // write out data in the second step
    source.append("                 if ((s_cl_helper[ai - 1] != 1) || (s_cl_helper[bi] != 1))  \n");
    source.append("                 {  \n");

                    // check how many childs are non terminated
    source.append("                     if (s_cl_helper[ai - 1] == 1)  \n");
    source.append("                     {  \n");
                        // mark as terminated
    source.append("                         s_cl_helper[bi] = 1;  \n");
    source.append("                     }  \n");
    source.append("                     else if (s_cl_helper[bi] == 1)  \n");
    source.append("                     {  \n");
                        // mark as terminated
    source.append("                         s_cl_helper[ai - 1] = 1;  \n");
    source.append("                     }  \n");
    source.append("               else      \n");   // both childs are non-terminated
    source.append("                     {  \n");

    source.append("                         unsigned int temp = s_cl_blocking[bi] + s_cl_blocking[ai - 1];  \n");

    source.append("                         if (temp > MAX_THREADS_BLOCK)  \n");
    source.append("                         {  \n");

                            // the two child trees have to form separate blocks, terminate trees
    source.append("                             s_cl_helper[ai - 1] = 1;  \n");
    source.append("                             s_cl_helper[bi] = 1;  \n");
    source.append("                         }  \n");
    source.append("                         else  \n");
    source.append("                         {  \n");
                            // build up tree by joining subtrees
    source.append("                             s_cl_blocking[bi] = temp;  \n");
    source.append("                             s_cl_blocking[ai - 1] = 0;  \n");
    source.append("                         }  \n");
    source.append("                     }  \n");
    source.append("            } \n"); // end s_cl_helper update

    source.append("             }  \n");

    source.append("             offset <<= 1;  \n");
    source.append("         }  \n");


        // traverse down tree, this only for stream compaction, not for block
        // construction
    source.append("         for (int d = 2; d < num_threads_compaction; d <<= 1)  \n");
    source.append("         {  \n");

    source.append("             offset >>= 1;  \n");
    source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

            //
    source.append("             if (tid < (d-1))  \n");
    source.append("             {  \n");

    source.append("                 unsigned int  ai = offset*(tid+1) - 1;  \n");
    source.append("                 unsigned int  bi = ai + (offset >> 1);  \n");

    source.append("                 s_cl_one[bi] = s_cl_one[bi] + s_cl_one[ai];  \n");
    source.append("                 s_cl_mult[bi] = s_cl_mult[bi] + s_cl_mult[ai];  \n");
    source.append("             }  \n");
    source.append("         }  \n");

    source.append("     }  \n");

} 


////////////////////////////////////////////////////////////////////////////////
//! Store all non-empty intervals resulting from the subdivision of the interval
//! currently processed by the thread
////////////////////////////////////////////////////////////////////////////////
 

void generate_storeNonEmptyIntervalsLarge(StringType & source, std::string const & numeric_string)
{
    source.append("       \n");
    source.append("     void  \n");
    source.append("     storeNonEmptyIntervalsLarge(unsigned int addr,  \n");
    source.append("                                 const unsigned int num_threads_active,  \n");
    source.append("                                 float  *s_left, float *s_right,  \n");
    source.append("                                 unsigned short  *s_left_count,  \n");
    source.append("                                 unsigned short *s_right_count,  \n");
    source.append("                                 float left, float mid, float right,  \n");
    source.append("                                 const unsigned short left_count,  \n");
    source.append("                                 const unsigned short mid_count,  \n");
    source.append("                                 const unsigned short right_count,  \n");
    source.append("                                 float epsilon,  \n");
    source.append("                                 unsigned int &compact_second_chunk,  \n");
    source.append("                                 unsigned short *s_compaction_list,  \n");
    source.append("                                 unsigned int &is_active_second)  \n");
    source.append("     {  \n");
    source.append("         uint glb_id = get_global_id(0); \n");
    source.append("         uint grp_id = get_group_id(0); \n");
    source.append("         uint grp_nm = get_num_groups(0); \n");
    source.append("         uint lcl_id = get_local_id(0); \n");
    source.append("         uint lcl_sz = get_local_size(0); \n");

        // check if both child intervals are valid
    source.append("         if ((left_count != mid_count) && (mid_count != right_count))  \n");
    source.append("         {  \n");

    source.append("             storeInterval(addr, s_left, s_right, s_left_count, s_right_count,  \n");
    source.append("                           left, mid, left_count, mid_count, epsilon);  \n");

    source.append("             is_active_second = 1;  \n");
    source.append("             s_compaction_list[lcl_id] = 1;  \n");
    source.append("             compact_second_chunk = 1;  \n");
    source.append("         }  \n");
    source.append("         else  \n");
    source.append("         {  \n");

            // only one non-empty child interval

            // mark that no second child
    source.append("             is_active_second = 0;  \n");
    source.append("             s_compaction_list[lcl_id] = 0;  \n");

            // store the one valid child interval
    source.append("             if (left_count != mid_count)  \n");
    source.append("             {  \n");
    source.append("                 storeInterval(addr, s_left, s_right, s_left_count, s_right_count,  \n");
    source.append("                               left, mid, left_count, mid_count, epsilon);  \n");
    source.append("             }  \n");
    source.append("             else  \n");
    source.append("             {  \n");
    source.append("                 storeInterval(addr, s_left, s_right, s_left_count, s_right_count,  \n");
    source.append("                               mid, right, mid_count, right_count, epsilon);  \n");
    source.append("             }  \n");
    source.append("         }  \n");
    source.append("     }  \n");
  //}
//}
#endif // #ifndef _BISECT_KERNEL_LARGE_H_

} 


 
