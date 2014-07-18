#ifndef VIENNACL_LINALG_CUDA_MATRIX_OPERATIONS_COL_HPP_
#define VIENNACL_LINALG_CUDA_MATRIX_OPERATIONS_COL_HPP_

/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file  viennacl/linalg/cuda/matrix_operations_col.hpp
    @brief Implementations of column-major dense matrix related operations, including matrix-vector products, using CUDA.
*/


namespace viennacl
{
  namespace linalg
  {
    namespace cuda
    {
      //
      // am
      //

      // alpha on CPU
      template <typename T>
      __global__ void am_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                T fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        T alpha = fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        if (options2 & (1 << 1))
        {
          for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
            for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
              A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha;
        }
        else
        {
          for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
            for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
              A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha;
        }
      }

      // alpha on GPU
      template <typename T>
      __global__ void am_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        T alpha = *fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

        if (options2 & (1 << 1))
        {
          for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
            for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
              A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha;
        }
        else
        {
          for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
            for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
              A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha;
        }
      }


      //
      // ambm
      //

      // alpha and beta on CPU
      template <typename T>
      __global__ void ambm_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                T fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,

                T fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
      {
        T alpha = fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;

        T beta = fac3;
        if (options3 & (1 << 0))
          beta = -beta;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

        if (options2 & (1 << 1))
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
        else
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
      }


      // alpha on CPU, beta on GPU
      template <typename T>
      __global__ void ambm_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                T fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,

                const T * fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
      {
        T alpha = fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;

        T beta = *fac3;
        if (options3 & (1 << 0))
          beta = -beta;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

        if (options2 & (1 << 1))
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
        else
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
      }

      // alpha on GPU, beta on CPU
      template <typename T>
      __global__ void ambm_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,

                T fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
      {
        T alpha = *fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;

        T beta = fac3;
        if (options3 & (1 << 0))
          beta = -beta;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

        if (options2 & (1 << 1))
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
        else
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
      }


      // alpha and beta on GPU
      template <typename T>
      __global__ void ambm_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,

                const T * fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
      {
        T alpha = *fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;

        T beta = *fac3;
        if (options3 & (1 << 0))
          beta = -beta;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

        if (options2 & (1 << 1))
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
        else
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
      }


      //
      // ambm_m
      //

      // alpha and beta on CPU
      template <typename T>
      __global__ void ambm_m_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                T fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,

                T fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
      {
        T alpha = fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;

        T beta = fac3;
        if (options3 & (1 << 0))
          beta = -beta;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

        if (options2 & (1 << 1))
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
             += B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
             += B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
        else
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
             += B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
             += B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
      }


      // alpha on CPU, beta on GPU
      template <typename T>
      __global__ void ambm_m_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                T fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,

                const T * fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
      {
        T alpha = fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;

        T beta = *fac3;
        if (options3 & (1 << 0))
          beta = -beta;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

        if (options2 & (1 << 1))
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
        else
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
      }

      // alpha on GPU, beta on CPU
      template <typename T>
      __global__ void ambm_m_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,

                T fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
      {
        T alpha = *fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;

        T beta = fac3;
        if (options3 & (1 << 0))
          beta = -beta;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

        if (options2 & (1 << 1))
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
        else
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
      }


      // alpha and beta on GPU
      template <typename T>
      __global__ void ambm_m_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,

                const T * fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
      {
        T alpha = *fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;

        T beta = *fac3;
        if (options3 & (1 << 0))
          beta = -beta;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

        if (options2 & (1 << 1))
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
        else
        {
          if (options3 & (1 << 1))
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
          }
          else
          {
            for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
              for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
                A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
              = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
              + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
          }
        }
      }



      //
      // assignments
      //

      template <typename T>
      __global__ void matrix_col_assign_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,
                T alpha)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = alpha;
      }


      template <typename T>
      __global__ void matrix_col_diagonal_assign_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,
                T alpha)
      {
        unsigned int gid = (blockIdx.x * blockDim.x + threadIdx.x);

        for (unsigned int row = gid; row < A_size1; row += blockDim.x * gridDim.x)
          A[(row * A_inc1 + A_start1) + (row * A_inc2 + A_start2) * A_internal_size1] = alpha;
      }

      //
      // binary element-wise operations
      //

      template <typename T>
      __global__ void element_op_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,

                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2,

                unsigned int op_type) //0: product, 1: division, 2: pow
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

        if (op_type == 2)
        {
          for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
            for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
              A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
            = pow(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1],
                  C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1]);
        }
        else if (op_type == 1)
        {
          for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
            for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
              A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
            = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]
            / C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1];
        }
        else if (op_type == 0)
        {
          for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
            for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
              A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
            = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]
            * C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1];
        }
      }

      template <typename T>
      __global__ void element_op_int_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,

                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2,

                unsigned int op_type) //0: product, 1: division, 2: pow
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

        if (op_type == 1)
        {
          for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
            for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
              A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
            = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]
            / C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1];
        }
        else if (op_type == 0)
        {
          for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
            for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
              A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
            = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]
            * C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1];
        }
      }


      //
      // unary element-wise operations
      //

      // abs
      template <typename T>
      __global__ void matrix_col_element_abs_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = abs(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // acos
      template <typename T>
      __global__ void matrix_col_element_acos_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = acos(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // asin
      template <typename T>
      __global__ void matrix_col_element_asin_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = asin(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // atan
      template <typename T>
      __global__ void matrix_col_element_atan_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = atan(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // ceil
      template <typename T>
      __global__ void matrix_col_element_ceil_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = ceil(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // cos
      template <typename T>
      __global__ void matrix_col_element_cos_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = cos(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // cosh
      template <typename T>
      __global__ void matrix_col_element_cosh_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = cosh(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // exp
      template <typename T>
      __global__ void matrix_col_element_exp_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = exp(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // fabs
      template <typename T>
      __global__ void matrix_col_element_fabs_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = fabs(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // floor
      template <typename T>
      __global__ void matrix_col_element_floor_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = floor(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // log
      template <typename T>
      __global__ void matrix_col_element_log_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = log(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // log10
      template <typename T>
      __global__ void matrix_col_element_log10_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = log10(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // sin
      template <typename T>
      __global__ void matrix_col_element_sin_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = sin(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // sinh
      template <typename T>
      __global__ void matrix_col_element_sinh_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = sinh(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // sqrt
      template <typename T>
      __global__ void matrix_col_element_sqrt_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = sqrt(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // tan
      template <typename T>
      __global__ void matrix_col_element_tan_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = tan(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }


      // tanh
      template <typename T>
      __global__ void matrix_col_element_tanh_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      {
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = tanh(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
      }



      //
      // matrix-vector product
      //

      template <typename T>
      __global__ void vec_mul_col_kernel(
                const T * A,
                unsigned int A_row_start,
                unsigned int A_col_start,
                unsigned int A_row_inc,
                unsigned int A_col_inc,
                unsigned int A_row_size,
                unsigned int A_col_size,
                unsigned int A_internal_rows,
                unsigned int A_internal_cols,
                const T * v,
                unsigned int v_start,
                unsigned int v_inc,
                unsigned int v_size,
                T * result,
                unsigned int result_start,
                unsigned int result_inc,
                unsigned int result_size)
      {

        for (unsigned int row = blockIdx.x * blockDim.x + threadIdx.x; row < A_row_size; row += gridDim.x * blockDim.x)
        {
          T dot_prod = 0;
          for (unsigned int col = 0; col < A_col_size; ++col)
            dot_prod += A[(row * A_row_inc + A_row_start) + (col * A_col_inc + A_col_start) * A_internal_rows] * v[v_start + v_inc * col];
          result[row * result_inc + result_start] = dot_prod;
        }
      }


      template <typename T>
      __global__ void trans_vec_mul_col_kernel(
                const T * A,
                unsigned int A_row_start,
                unsigned int A_col_start,
                unsigned int A_row_inc,
                unsigned int A_col_inc,
                unsigned int A_row_size,
                unsigned int A_col_size,
                unsigned int A_internal_rows,
                unsigned int A_internal_cols,
                const T * v,
                unsigned int v_start,
                unsigned int v_inc,
                unsigned int v_size,
                T * result,
                unsigned int result_start,
                unsigned int result_inc,
                unsigned int result_size)
      {
        __shared__ T work[128];

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int lid = threadIdx.x;

        for (unsigned int row = row_gid; row < A_col_size; row += gridDim.x)
        {
          T dot_prod = 0;
          for (unsigned int col = col_gid; col < A_row_size; col += blockDim.x)
            dot_prod += A[(row * A_col_inc + A_col_start) * A_internal_rows + col * A_row_inc + A_row_start] * v[v_start + v_inc * col];
          work[lid] = dot_prod;

          for(unsigned int stride = blockDim.x/2 ; stride>0 ; stride>>=1){
            __syncthreads();
            if(lid < stride)
              work[lid] += work[lid+stride];
          }

          if(lid == 0)
            result[row * result_inc + result_start] = work[0];
        }
      }


      //
      // matrix-matrix products
      //




      //
      // scaled rank-1-update
      //

      // alpha on CPU
      template <typename T>
      __global__ void scaled_rank1_update_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                T val,
                unsigned int options2,

                const T * vec1,
                unsigned int start1,
                unsigned int inc1,
                unsigned int size1,

                const T * vec2,
                unsigned int start2,
                unsigned int inc2,
                unsigned int size2)
      {
        T alpha = val;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        {
          T tmp = alpha * vec1[row * inc1 + start1];
          for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] += tmp * vec2[col * inc2 + start2];
        }
      }


      // alpha on GPU
      template <typename T>
      __global__ void scaled_rank1_update_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                const T * val,
                unsigned int options2,

                const T * vec1,
                unsigned int start1,
                unsigned int inc1,
                unsigned int size1,

                const T * vec2,
                unsigned int start2,
                unsigned int inc2,
                unsigned int size2)
      {
        T alpha = *val;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

        for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        {
          T tmp = alpha * vec1[row * inc1 + start1];
          for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] += tmp * vec2[col * inc2 + start2];
        }
      }

      // alpha on GPU
      template <typename T>
      __global__ void bidiag_pack_kernel(
                  T * A,
                  T * D,
                  T * S,
                  uint size1,
                  uint size2,
                  uint stride)
     {
        uint size = min(size1, size2);
        if(blockIdx.x * blockDim.x + threadIdx.x == 0)
          S[0] = 0;

        for(uint i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < size;
                 i += gridDim.x * blockDim.x)
          {
            D[i] = A[i*stride + i];
            S[i+1] = (i + 1 < size2) ? A[i*stride + (i + 1)] : 0;
          }
      }


      template <typename NumericT, typename F, typename VectorType>
      void bidiag_pack(matrix_base<NumericT, F> & A,
                       VectorType & dh,
                       VectorType & sh
                      )
      {
        std::cout << "Bidiag_pack started in CUDA !!\n";
        viennacl::vector<NumericT> D(dh.size());
        viennacl::vector<NumericT> S(sh.size());

	viennacl::linalg::cuda::bidiag_pack_kernel<<<128, 128>>>(detail::cuda_arg<NumericT>(A),
								 viennacl::linalg::cuda::detail::cuda_arg<NumericT>(D),
								 viennacl::linalg::cuda::detail::cuda_arg<NumericT>(S),
								 static_cast<unsigned int>(viennacl::traits::size1(A)),
								 static_cast<unsigned int>(viennacl::traits::size2(A)),
								 static_cast<unsigned int>(viennacl::traits::internal_size2(A)));

        fast_copy(D, dh);
        fast_copy(S, sh);
      }




      template<typename T>
      __global__ void copy_col_kernel(
              T * A,
              T * V,
              uint row_start,
              uint col_start,
              uint size,
              uint stride)
      {
          uint x = blockIdx.x * blockDim.x + threadIdx.x;
          uint sz = gridDim.x * blockDim.x;

          for(uint i = row_start + x; i < size; i += sz)
          {
              V[i - row_start] = A[i * stride + col_start];
          }
      }

      template<typename T>
      __global__ void copy_row_kernel(
              T * A,
              T * V,
              uint row_start,
              uint col_start,
              uint size,
              uint stride)
      {
          uint x = blockIdx.x * blockDim.x + threadIdx.x;
          uint sz = gridDim.x * blockDim.x;

          for(uint i = col_start + x; i < size; i += sz)
          {
              V[i - col_start] = A[row_start * stride + i];
          }

      }

      template <typename NumericT, typename F>
      void copy_vec(matrix_base<NumericT, F>& A,
                    vector_base<NumericT> & V,
                    vcl_size_t row_start,
                    vcl_size_t col_start,
                    bool copy_col
      )
      {
        if(copy_col)
          {
            copy_col_kernel<<<128, 128>>>(detail::cuda_arg<NumericT>(A),
                                          detail::cuda_arg<NumericT>(V),
                                          static_cast<unsigned int>(row_start),         // vcl_size_t oder unsigned int ?
                                          static_cast<unsigned int>(col_start),
                                          static_cast<unsigned int>(viennacl::traits::size1(A)),
                                          static_cast<unsigned int>(viennacl::traits::internal_size2(A)));

          }
        else
          {
            copy_row_kernel<<<128, 128>>>(detail::cuda_arg<NumericT>(A),
                                          detail::cuda_arg<NumericT>(V),
                                          static_cast<unsigned int>(row_start),
                                          static_cast<unsigned int>(col_start),
                                          static_cast<unsigned int>(viennacl::traits::size2(A)),
                                          static_cast<unsigned int>(viennacl::traits::internal_size2(A)));
          }
      }


      template<typename T>
      __global__ void house_update_A_left_kernel(
              T * A,
              T * V,        //housholder vector
              uint row_start,
              uint col_start,
              uint size1,
              uint size2,
              uint stride)
      {
         // __shared__ T sums[128];  //unused variable
          T ss = 0;

          for(uint i = blockIdx.x * blockDim.x + threadIdx.x + col_start;
              i < size2;
              i += gridDim.x * blockDim.x)
          {
              ss = 0;
              for(uint j = row_start; j < size1; j++)
                  ss = ss +(V[j] * A[j * stride + i]);

              for(uint j = row_start; j < size1; j++)
                  A[j * stride + i] = A[j * stride + i] - (2 * V[j] * ss);
          }
      }

      template <typename NumericT, typename F>
      void house_update_A_left(matrix_base<NumericT, F> & A,
                               vector_base<NumericT> & D,
                               vcl_size_t start)
      {
        house_update_A_left_kernel<<<128, 128>>>(detail::cuda_arg<NumericT>(A),
                                                 detail::cuda_arg<NumericT>(D),
                                                 static_cast<unsigned int>(start + 1),
                                                 static_cast<unsigned int>(start),
                                                 static_cast<unsigned int>(viennacl::traits::size1(A)),
                                                 static_cast<unsigned int>(viennacl::traits::size2(A)),
                                                 static_cast<unsigned int>(viennacl::traits::internal_size2(A)));

      }



      template<typename T>
      __global__ void house_update_A_right_kernel(
              T * A,
              T * V,  //housholder vector
              uint row_start,
              uint col_start,
              uint size1,
              uint size2,
              uint stride)
      {
          __shared__ T sums[128];
          T ss = 0;

          for(uint i = blockIdx.x + row_start; i < size1; i+= gridDim.x)
          {
              ss = 0;
              for(uint j = threadIdx.x; j < size2; j+= blockDim.x)
                  ss = ss + (V[j] * A[i * stride + j]);
              //sums[threadIdx.x]; //no effect

              __syncthreads();
              col_reduce_lcl_array(sums, threadIdx.x, blockDim.x);
              __syncthreads();

              T sum_Av = sums[0];

              for(uint j = threadIdx.x; j < size2; j+= blockDim.x)
                  A[i * stride + j] = A[i * stride + j] - (2 * V[j] * sum_Av);
          }
      }

      template <typename NumericT, typename F>
      void house_update_A_right(matrix_base<NumericT, F> & A,
                               vector_base<NumericT> & D,
                               vcl_size_t start)
      {
        house_update_A_right_kernel<<<128, 128>>>(detail::cuda_arg<NumericT>(A),
                                                  detail::cuda_arg<NumericT>(D),
                                                  static_cast<unsigned int>(0),
                                                  static_cast<unsigned int>(0),
                                                  static_cast<unsigned int>(viennacl::traits::size1(A)),
                                                  static_cast<unsigned int>(viennacl::traits::size2(A)),
                                                  static_cast<unsigned int>(viennacl::traits::internal_size2(A)));

      }


      template<typename T>
      __device__ void col_reduce_lcl_array(
              T * sums,
              uint th_Idx,
              uint bl_Dim)
      {
          uint step = bl_Dim >> 1;

          while(step > 0)
          {
              if(th_Idx < step)
                  sums[th_Idx] += sums[th_Idx + step];
              step >>= 1;
              __syncthreads();
          }
      }


      template <typename T>
      __global__ void house_update_QL_kernel(
              T * QL,
              T * V,
              uint size1,
              uint size2,
              uint strideQ)
      {
          __shared__ T sums[128];
          T ss = 0;
          for(uint i = blockIdx.x; i < size1; i += gridDim.x)
          {
              ss = 0;
              for(uint j = threadIdx.x; j < size1; j += blockDim.x)
                  ss = ss + (V[j] * QL[i * strideQ + j]);
              sums[threadIdx.x] = ss;

              __syncthreads();
              col_reduce_lcl_array(sums, threadIdx.x, blockDim.x);
              __syncthreads();

              T sum_Qv = sums[0];

              for(uint j = threadIdx.x; j < size1; j += blockDim.x)
                  QL[i * strideQ + j] = QL[i * strideQ + j] - (2 * V[j] * sum_Qv);
          }
      }


      template <typename NumericT, typename F>
      void house_update_QL(matrix_base<NumericT, F> & A,
                           matrix_base<NumericT, F> & Q,
                           vector_base<NumericT> & D)

      {
        house_update_QL_kernel<<<128, 128>>>(detail::cuda_arg<NumericT>(Q),
                                             detail::cuda_arg<NumericT>(D),
                                             static_cast<unsigned int>(viennacl::traits::size1(A)),
                                             static_cast<unsigned int>(viennacl::traits::size2(A)),
                                             static_cast<unsigned int>(viennacl::traits::internal_size2(Q)));
      }

      template <typename T>
      __global__ void givens_next_kernel(
              T * matr,
              T * cs,
              T * ss,
              uint size,
              uint stride,
              uint start_i,
              uint end_i)
      {
          uint j = blockIdx.x * blockDim.x + threadIdx.x;
          __shared__ T cs_lcl[256];
          __shared__ T ss_lcl[256];

          T x = (j < size) ? matr[(end_i + 1) * stride + j] : 0;

          uint elems_num = end_i - start_i + 1;
          uint block_num = (elems_num + blockDim.x - 1) / blockDim.x;

          for(uint block_id = 0; block_id < block_num; block_id++)
          {
              uint to = min(elems_num - block_id * blockDim.x, blockDim.x);

              if(threadIdx.x < to)
              {
                  cs_lcl[threadIdx.x] = cs[end_i - (threadIdx.x + block_id * threadIdx.x)];
                  ss_lcl[threadIdx.x] = ss[end_i - (threadIdx.x + block_id * threadIdx.x)];
              }
              __syncthreads();
              if(j < size)
              {
                  for(uint ind = 0; ind < to; ind++)
                  {
                      uint i = end_i - (ind + block_id * threadIdx.x);
                      T z = matr[i * stride + j];
                      T cs_val = cs_lcl[ind];
                      T ss_val = ss_lcl[ind];
                      matr[(i + 1) * stride + j] = x * cs_val + z * ss_val;
                      x = -x * ss_val + z * cs_val;
                  }
              }
              __syncthreads();
           }
           if(j < size)
             matr[(start_i) * stride + j] = x;
      }
      template<typename NumericT, typename F>
        void givens_next(matrix_base<NumericT, F> & matrix,
                        vector_base<NumericT>& tmp1,
                        vector_base<NumericT>& tmp2,
                        int l,
                        int m)
        {
        givens_next_kernel<<<128, 128>>>(detail::cuda_arg<NumericT>(matrix),
                                         detail::cuda_arg<NumericT>(tmp1),
                                         detail::cuda_arg<NumericT>(tmp2),
                                         static_cast<unsigned int>(viennacl::traits::size1(matrix)),
                                         static_cast<unsigned int>(viennacl::traits::internal_size2(matrix)),
                                         static_cast<unsigned int>(l),
                                         static_cast<unsigned int>(m - 1));
      }

    } // namespace cuda
  } //namespace linalg
} //namespace viennacl


#endif
