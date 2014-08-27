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

/* Helper structures to simplify variable handling */

#ifndef _STRUCTS_H_
#define _STRUCTS_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/eigenvalues/util.hpp"

class InputData
{
  public:
  ////////////////////////////////////////////////////////////////////////////////
  //! Initialize the input data to the algorithm
  //! @param input  handles to the input data
  //! @param exec_path  path where executable is run (argv[0])
  //! @param mat_size  size of the matrix
  //! @param user_defined  1 if the matrix size has been requested by the user,
  //!                      0 if the default size
  ////////////////////////////////////////////////////////////////////////////////

  InputData(char *exec_path, const unsigned int sz, const unsigned int user_defined) :
              std_a(sz), std_b(sz),  std_b_raw(sz), vcl_a(sz)
    {
        // allocate memory
      const unsigned int mat_size = sz;
      a = (float *) malloc(sizeof(float) * mat_size);
      b = (float *) malloc(sizeof(float) * mat_size);
      
      std::cout << "Init Input Data!" << std::endl;

      if (0 == user_defined)
      {
          /*
           std_a[0] = 1;  std_b_raw[0] = 0;
           std_a[1] = 2;  std_b_raw[1] = 4;
           std_a[2] =-4;  std_b_raw[2] = 5;
           std_a[3] = 6;  std_b_raw[3] = 1;
           std_a[4] = 3;  std_b_raw[4] = 2;
           std_a[5] = 4;  std_b_raw[5] =-3;
           std_a[6] = 7;  std_b_raw[6] = 5;
           std_a[7] = 9;  std_b_raw[7] = 1;
           std_a[8] = 3;  std_b_raw[8] = 5;
           std_a[9] = 8;  std_b_raw[9] = 2;

           */
            srand(278217421);
            /*
           for(unsigned int i = 0; i < mat_size; ++i)
           {
             //std_a[i] = i % 11 + 4;
             //std_b_raw[i] = i % 9 + 2;
             
             a[i] = ((float)(i % 9)) - 4.5f;
             b[i] = ((float)(i % 5)) - 4.5f;

           }*/
           
           // initialize diagonal and superdiagonal entries with random values
       

        // srand( clock());
        
        for (unsigned int i = 0; i < mat_size; ++i)
        {
            a[i] = (double)(12.0 * (((double)rand()
                                         / (float) RAND_MAX) - 0.5));
            b[i] = (double)(12.0 * (((double)rand()
                                         / (float) RAND_MAX) - 0.5));
        }

          // the first element of s is used as padding on the device (thus the
          // whole vector is copied to the device but the kernels are launched
          // with (s+1) as start address
           //std_b_raw[0] = 0.0f;
          b[0] = 0.0f;
      }


      // allocate device memory for input
      checkCudaErrors(cudaMalloc((void **) &( g_a)    , sizeof(float) * mat_size));
      checkCudaErrors(cudaMalloc((void **) &( g_b_raw), sizeof(float) * mat_size));

      // copy data to device
      //copy(std_a, vcl_a);
     // copy(std_b_raw, vcl_b_raw);
      
     /* copy(std_b_raw.begin() + 0,  std_b_raw.end(),  std_b.begin());

      copy(std_a.begin(), std_a.end(), a);
      copy(std_b.begin(), std_b.end(), b);
     */
      checkCudaErrors(cudaMemcpy(g_a    , a, sizeof(float) * mat_size, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(g_b_raw, b, sizeof(float) * mat_size, cudaMemcpyHostToDevice));


       g_b =  g_b_raw + 1;
    }


    ////////////////////////////////////////////////////////////////////////////////
    //! Clean up input data, in particular allocated memory
    //! @param input  handles to the input data
    ////////////////////////////////////////////////////////////////////////////////
    void
    cleanupInputData(void)
    {

        freePtr(a);
        freePtr(b);

        checkCudaErrors(cudaFree(g_a));
        g_a = NULL;
        checkCudaErrors(cudaFree(g_b_raw));
        g_b_raw = NULL;
        g_b = NULL;

    }
    
    
    //! host/device side representation of diagonal
    float  *a;
    viennacl::vector<float> vcl_a;
    std::vector<float> std_a;
    //! host/device side representation superdiagonal
    //viennacl::vector<float> vcl_b;
    std::vector<float> std_b;
    //! host/device side representation of helper vector
    //viennacl::vector<float> vcl_b_raw;
    std::vector<float> std_b_raw;
    
    //! host side representation superdiagonal
    float  *b;

    //! device side representation of diagonal
    float  *g_a;
    //! device side representation of superdiagonal
    float  *g_b;
    //! helper variable pointing to the mem allocated for g_b which provides
    //! space for one additional element of padding at the beginning
    float  *g_b_raw;

};




class ResultDataLarge
{
public:
    ResultDataLarge(unsigned int sz) //: std_eigenvalues(sz)
    {
    }
    
    //! eigenvalues
   // std::vector<float> std_eigenvalues;
    
    //! number of intervals containing one eigenvalue after the first step
    unsigned int *g_num_one;

    //! number of (thread) blocks of intervals containing multiple eigenvalues
    //! after the first steo
    unsigned int *g_num_blocks_mult;

    //! left interval limits of intervals containing one eigenvalue after the
    //! first iteration step
    float *g_left_one;

    //! right interval limits of intervals containing one eigenvalue after the
    //! first iteration step
    float *g_right_one;

    //! interval indices (position in sorted listed of eigenvalues)
    //! of intervals containing one eigenvalue after the first iteration step
    unsigned int *g_pos_one;

    //! left interval limits of intervals containing multiple eigenvalues
    //! after the first iteration step
    float *g_left_mult;

    //! right interval limits of intervals containing multiple eigenvalues
    //! after the first iteration step
    float *g_right_mult;

    //! number of eigenvalues less than the left limit of the eigenvalue
    //! intervals containing multiple eigenvalues
    unsigned int *g_left_count_mult;

    //! number of eigenvalues less than the right limit of the eigenvalue
    //! intervals containing multiple eigenvalues
    unsigned int *g_right_count_mult;

    //! start addresses in g_left_mult etc. of blocks of intervals containing
    //! more than one eigenvalue after the first step
    unsigned int  *g_blocks_mult;

    //! accumulated number of intervals in g_left_mult etc. of blocks of
    //! intervals containing more than one eigenvalue after the first step
    unsigned int  *g_blocks_mult_sum;

    //! eigenvalues that have been generated in the second step from intervals
    //! that still contained multiple eigenvalues after the first step
    float *g_lambda_mult;

    //! eigenvalue index of intervals that have been generated in the second
    //! processing step
    unsigned int *g_pos_mult;

};

#endif // #ifndef _STRUCTS_H_

