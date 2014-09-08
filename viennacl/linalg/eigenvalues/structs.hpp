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

//namespace viennacl
//{
  //namespace linalg
  //{
    class InputData
    {
      public:

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
      //viennacl::vector<float> g_a;
      float  *g_a;
      //! device side representation of superdiagonal
      float  *g_b;
      //! helper variable pointing to the mem allocated for g_b which provides
      //! space for one additional element of padding at the beginning
      float  *g_b_raw;


      ////////////////////////////////////////////////////////////////////////////////
      //! Initialize the input data to the algorithm
      //! @param input  handles to the input data
      //! @param mat_size  size of the matrix
      ////////////////////////////////////////////////////////////////////////////////

      InputData(std::vector<float> diagonal, std::vector<float> superdiagonal, const unsigned int sz) :
                  std_a(sz), std_b(sz),  std_b_raw(sz)//, g_a(sz)
      {
        // allocate memory
        const unsigned int mat_size = sz;
        a = (float *) malloc(sizeof(float) * mat_size);
        b = (float *) malloc(sizeof(float) * mat_size);

       std::copy(diagonal.begin(), diagonal.end(), a);
       std::copy(superdiagonal.begin(), superdiagonal.end(), b);

       // the first element of s is used as padding on the device (thus the
       // whole vector is copied to the device but the kernels are launched
       // with (s+1) as start address
       b[0] = 0.0f;

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
      // copy(diagonal.begin(), diagonal.end(), g_a);
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

    };


    class ResultDataSmall
    {
    public:
      //! eigenvalues (host side)
      std::vector<float> std_eigenvalues;


      // left interval limits at the end of the computation
      float *g_left;
      viennacl::vector<float> vcl_g_left;

      // right interval limits at the end of the computation
      float *g_right;

      // number of eigenvalues smaller than the left interval limit
      unsigned int *g_left_count;

      // number of eigenvalues bigger than the right interval limit
      unsigned int *g_right_count;

      //! flag if algorithm converged
      unsigned int *g_converged;

      // helper variables

      unsigned int mat_size_f;
      unsigned int mat_size_ui;

      float         *zero_f;
      unsigned int  *zero_ui;


      ////////////////////////////////////////////////////////////////////////////////
      //! Initialize variables and memory for the result for small matrices
      ////////////////////////////////////////////////////////////////////////////////
      ResultDataSmall(const unsigned int mat_size) : std_eigenvalues(mat_size), vcl_g_left(mat_size)
      {

          mat_size_f = sizeof(float) * mat_size;
          mat_size_ui = sizeof(unsigned int) * mat_size;

          //eigenvalues = (float *) malloc(mat_size_f);

          // helper variables
          zero_f = (float *) malloc(mat_size_f);
          zero_ui = (unsigned int *) malloc(mat_size_ui);

          for (unsigned int i = 0; i < mat_size; ++i)
          {

              zero_f[i] = 0.0f;
              zero_ui[i] = 0;
              vcl_g_left.clear();
          }

          checkCudaErrors(cudaMalloc((void **) &g_left, mat_size_f));
          checkCudaErrors(cudaMalloc((void **) &g_right, mat_size_f));

          checkCudaErrors(cudaMalloc((void **) &g_left_count,
                                     mat_size_ui));
          checkCudaErrors(cudaMalloc((void **) &g_right_count,
                                     mat_size_ui));

          // initialize result memory
          checkCudaErrors(cudaMemcpy(g_left, zero_f, mat_size_f,
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(g_right, zero_f, mat_size_f,
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(g_right_count, zero_ui,
                                     mat_size_ui,
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(g_left_count, zero_ui,
                                     mat_size_ui,
                                     cudaMemcpyHostToDevice));
      }

      ////////////////////////////////////////////////////////////////////////////////
      //! Cleanup memory and variables for result for small matrices
      ////////////////////////////////////////////////////////////////////////////////
      void
      cleanup()
      {

          freePtr(zero_f);
          freePtr(zero_ui);

          checkCudaErrors(cudaFree(g_left));
          checkCudaErrors(cudaFree(g_right));
          checkCudaErrors(cudaFree(g_left_count));
          checkCudaErrors(cudaFree(g_right_count));
      }
    };

    /////////////////////////////////////////////////////////////////////////////////
    //! In this class the all data of the result is stored
    /////////////////////////////////////////////////////////////////////////////////

    class ResultDataLarge
    {
    public:

        //! eigenvalues
        std::vector<float> std_eigenvalues;

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



        ////////////////////////////////////////////////////////////////////////////////
        //! Initialize variables and memory for result
        //! @param  result handles to memory
        //! @param  matrix_size  size of the matrix
        ////////////////////////////////////////////////////////////////////////////////
        ResultDataLarge(const unsigned int mat_size) : std_eigenvalues(mat_size)
        {

            // helper variables to initialize memory
            unsigned int zero = 0;
            unsigned int mat_size_f = sizeof(float) * mat_size;
            unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;

            float *tempf = (float *) malloc(mat_size_f);
            unsigned int *tempui = (unsigned int *) malloc(mat_size_ui);

            for (unsigned int i = 0; i < mat_size; ++i)
            {
                tempf[i] = 0.0f;
                tempui[i] = 0;
            }

            // number of intervals containing only one eigenvalue after the first step
            checkCudaErrors(cudaMalloc((void **)  &g_num_one,
                                       sizeof(unsigned int)));
            checkCudaErrors(cudaMemcpy(g_num_one, &zero, sizeof(unsigned int),
                                       cudaMemcpyHostToDevice));

            // number of (thread) blocks of intervals with multiple eigenvalues after
            // the first iteration
            checkCudaErrors(cudaMalloc((void **) & g_num_blocks_mult,
                                       sizeof(unsigned int)));
            checkCudaErrors(cudaMemcpy( g_num_blocks_mult, &zero,
                                       sizeof(unsigned int),
                                       cudaMemcpyHostToDevice));


            checkCudaErrors(cudaMalloc((void **) & g_left_one, mat_size_f));
            checkCudaErrors(cudaMalloc((void **) & g_right_one, mat_size_f));
            checkCudaErrors(cudaMalloc((void **) & g_pos_one, mat_size_ui));

            checkCudaErrors(cudaMalloc((void **) & g_left_mult, mat_size_f));
            checkCudaErrors(cudaMalloc((void **) & g_right_mult, mat_size_f));
            checkCudaErrors(cudaMalloc((void **) & g_left_count_mult,
                                       mat_size_ui));
            checkCudaErrors(cudaMalloc((void **) & g_right_count_mult,
                                       mat_size_ui));

            checkCudaErrors(cudaMemcpy( g_left_one, tempf, mat_size_f,
                                       cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy( g_right_one, tempf, mat_size_f,
                                       cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy( g_pos_one, tempui, mat_size_ui,
                                       cudaMemcpyHostToDevice));

            checkCudaErrors(cudaMemcpy( g_left_mult, tempf, mat_size_f,
                                       cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy( g_right_mult, tempf, mat_size_f,
                                       cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy( g_left_count_mult, tempui, mat_size_ui,
                                       cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy( g_right_count_mult, tempui, mat_size_ui,
                                       cudaMemcpyHostToDevice));

            checkCudaErrors(cudaMalloc((void **) & g_blocks_mult, mat_size_ui));
            checkCudaErrors(cudaMemcpy( g_blocks_mult, tempui, mat_size_ui,
                                       cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMalloc((void **) & g_blocks_mult_sum, mat_size_ui));
            checkCudaErrors(cudaMemcpy( g_blocks_mult_sum, tempui, mat_size_ui,
                                       cudaMemcpyHostToDevice));

            checkCudaErrors(cudaMalloc((void **) & g_lambda_mult, mat_size_f));
            checkCudaErrors(cudaMemcpy( g_lambda_mult, tempf, mat_size_f,
                                       cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMalloc((void **) & g_pos_mult, mat_size_ui));
            checkCudaErrors(cudaMemcpy( g_pos_mult, tempf, mat_size_ui,
                                       cudaMemcpyHostToDevice));

        }

        ////////////////////////////////////////////////////////////////////////////////
        //! Cleanup result memory
        //! @param result  handles to memory
        ////////////////////////////////////////////////////////////////////////////////
        void
        cleanup()
        {

            checkCudaErrors(cudaFree( g_num_one));
            checkCudaErrors(cudaFree( g_num_blocks_mult));
            checkCudaErrors(cudaFree( g_left_one));
            checkCudaErrors(cudaFree( g_right_one));
            checkCudaErrors(cudaFree( g_pos_one));
            checkCudaErrors(cudaFree( g_left_mult));
            checkCudaErrors(cudaFree( g_right_mult));
            checkCudaErrors(cudaFree( g_left_count_mult));
            checkCudaErrors(cudaFree( g_right_count_mult));
            checkCudaErrors(cudaFree( g_blocks_mult));
            checkCudaErrors(cudaFree( g_blocks_mult_sum));
            checkCudaErrors(cudaFree( g_lambda_mult));
            checkCudaErrors(cudaFree( g_pos_mult));
        }


    };
//  }
//}
#endif // #ifndef _STRUCTS_H_

