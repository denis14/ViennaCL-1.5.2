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

        //! host side representation of diagonal
        std::vector<float> std_a;
        //! host side representation superdiagonal
        std::vector<float> std_b;
        //! device side representation of diagonal
        viennacl::vector<float> vcl_a;
        //!device side representation of superdiagonal
        viennacl::vector<float> vcl_b;

        ////////////////////////////////////////////////////////////////////////////////
        //! Initialize the input data to the algorithm
        //! @param input  handles to the input data
        //! @param mat_size  size of the matrix
        ////////////////////////////////////////////////////////////////////////////////

        InputData(std::vector<float> diagonal, std::vector<float> superdiagonal, const unsigned int sz) :
                    std_a(sz), std_b(sz), vcl_a(sz), vcl_b(sz)
        {
         std_a = diagonal;
         std_b = superdiagonal;

         viennacl::copy(std_b, vcl_b);
         viennacl::copy(std_a, vcl_a);
      }
    };



    class ResultDataSmall
    {
      public:
        //! eigenvalues (host side)
        std::vector<float> std_eigenvalues;
        //! left interval limits at the end of the computation
        viennacl::vector<float> vcl_g_left;
        //! right interval limits at the end of the computation
        viennacl::vector<float> vcl_g_right;
        //! number of eigenvalues smaller than the left interval limit
        viennacl::vector<unsigned int> vcl_g_left_count;
        //! number of eigenvalues bigger than the right interval limit
        viennacl::vector<unsigned int> vcl_g_right_count;


        ////////////////////////////////////////////////////////////////////////////////
        //! Initialize variables and memory for the result for small matrices
        ////////////////////////////////////////////////////////////////////////////////
        ResultDataSmall(const unsigned int mat_size) :
          std_eigenvalues(mat_size), vcl_g_left(mat_size), vcl_g_right(mat_size), vcl_g_left_count(mat_size), vcl_g_right_count(mat_size)
        {
            vcl_g_left.clear();
            vcl_g_right.clear();
            vcl_g_left_count.clear();
            vcl_g_right_count.clear();
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
        viennacl::scalar<unsigned int> g_num_one;

        //! number of (thread) blocks of intervals containing multiple eigenvalues
        //! after the first steo
        viennacl::scalar<unsigned int> g_num_blocks_mult;

        //! left interval limits of intervals containing one eigenvalue after the
        //! first iteration step
        viennacl::vector<float> g_left_one;

        //! right interval limits of intervals containing one eigenvalue after the
        //! first iteration step
        viennacl::vector<float> g_right_one;

        //! interval indices (position in sorted listed of eigenvalues)
        //! of intervals containing one eigenvalue after the first iteration step
        viennacl::vector<unsigned int> g_pos_one;

        //! left interval limits of intervals containing multiple eigenvalues
        //! after the first iteration step
        viennacl::vector<float> g_left_mult;
        //! right interval limits of intervals containing multiple eigenvalues
        //! after the first iteration step
        viennacl::vector<float> g_right_mult;

        //! number of eigenvalues less than the left limit of the eigenvalue
        //! intervals containing multiple eigenvalues
        viennacl::vector<unsigned int> g_left_count_mult;

        //! number of eigenvalues less than the right limit of the eigenvalue
        //! intervals containing multiple eigenvalues
        viennacl::vector<unsigned int> g_right_count_mult;
        //! start addresses in g_left_mult etc. of blocks of intervals containing
        //! more than one eigenvalue after the first step
        viennacl::vector<unsigned int> g_blocks_mult;

        //! accumulated number of intervals in g_left_mult etc. of blocks of
        //! intervals containing more than one eigenvalue after the first step
        viennacl::vector<unsigned int> g_blocks_mult_sum;

        //! eigenvalues that have been generated in the second step from intervals
        //! that still contained multiple eigenvalues after the first step
        viennacl::vector<float> g_lambda_mult;

        //! eigenvalue index of intervals that have been generated in the second
        //! processing step
        viennacl::vector<unsigned int> g_pos_mult;



        ////////////////////////////////////////////////////////////////////////////////
        //! Initialize variables and memory for result
        //! @param  result handles to memory
        //! @param  matrix_size  size of the matrix
        ////////////////////////////////////////////////////////////////////////////////
        ResultDataLarge(const unsigned int mat_size) :
          std_eigenvalues(mat_size), g_left_one(mat_size), g_right_one(mat_size), g_pos_one(mat_size),
          g_left_mult(mat_size), g_right_mult(mat_size),g_left_count_mult(mat_size), g_right_count_mult(mat_size),
          g_lambda_mult(mat_size), g_blocks_mult(mat_size), g_blocks_mult_sum(mat_size), g_pos_mult(mat_size),
          g_num_one(0), g_num_blocks_mult(0)
        {
            g_left_one.clear();
            g_right_one.clear();
            g_pos_one.clear();
            g_left_mult.clear();
            g_right_mult.clear();
            g_left_count_mult.clear();
            g_right_count_mult.clear();
            g_lambda_mult.clear();
            g_blocks_mult.clear();
            g_blocks_mult_sum.clear();
            g_pos_mult.clear();
        }
    };
//  }
//}
#endif // #ifndef _STRUCTS_H_

