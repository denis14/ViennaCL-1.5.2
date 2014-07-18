#ifndef VIENNACL_LINALG_HOST_BASED_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_HOST_BASED_MATRIX_OPERATIONS_HPP_

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

/** @file  viennacl/linalg/host_based/matrix_operations.hpp
    @brief Implementations of dense matrix related operations, including matrix-vector products, using a plain single-threaded or OpenMP-enabled execution on CPU.
*/


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
#include "viennacl/linalg/detail/op_applier.hpp"
#include "viennacl/linalg/host_based/common.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/matrix_operations.hpp"

namespace viennacl
{
  namespace linalg
  {
    namespace host_based
    {

      //
      // Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
      //

      template <typename NumericT, typename F, typename ScalarType1>
      void am(matrix_base<NumericT, F> & mat1,
              matrix_base<NumericT, F> const & mat2, ScalarType1 const & alpha, vcl_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha)
      {
        typedef NumericT        value_type;

        value_type       * data_A = detail::extract_raw_pointer<value_type>(mat1);
        value_type const * data_B = detail::extract_raw_pointer<value_type>(mat2);

        value_type data_alpha = alpha;
        if (flip_sign_alpha)
          data_alpha = -data_alpha;

        vcl_size_t A_start1 = viennacl::traits::start1(mat1);
        vcl_size_t A_start2 = viennacl::traits::start2(mat1);
        vcl_size_t A_inc1   = viennacl::traits::stride1(mat1);
        vcl_size_t A_inc2   = viennacl::traits::stride2(mat1);
        vcl_size_t A_size1  = viennacl::traits::size1(mat1);
        vcl_size_t A_size2  = viennacl::traits::size2(mat1);
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat1);
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat1);

        vcl_size_t B_start1 = viennacl::traits::start1(mat2);
        vcl_size_t B_start2 = viennacl::traits::start2(mat2);
        vcl_size_t B_inc1   = viennacl::traits::stride1(mat2);
        vcl_size_t B_inc2   = viennacl::traits::stride2(mat2);
        vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(mat2);
        vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(mat2);

        detail::matrix_array_wrapper<value_type,       typename F::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename F::orientation_category, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        //typedef typename detail::majority_struct_for_orientation<typename M1::orientation_category>::type index_generator_A;
        //typedef typename detail::majority_struct_for_orientation<typename M2::orientation_category>::type index_generator_B;

        if (detail::is_row_major(typename F::orientation_category()))
        {
          if (reciprocal_alpha)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long row = 0; row < static_cast<long>(A_size1); ++row)
              for (long col = 0; col < static_cast<long>(A_size2); ++col)
                wrapper_A(row, col) = wrapper_B(row, col) / data_alpha;
          }
          else
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long row = 0; row < static_cast<long>(A_size1); ++row)
              for (long col = 0; col < static_cast<long>(A_size2); ++col)
                wrapper_A(row, col) = wrapper_B(row, col) * data_alpha;
          }
        }
        else
        {
          if (reciprocal_alpha)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long col = 0; col < static_cast<long>(A_size2); ++col)
              for (long row = 0; row < static_cast<long>(A_size1); ++row)
                wrapper_A(row, col) = wrapper_B(row, col) / data_alpha;
          }
          else
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long col = 0; col < static_cast<long>(A_size2); ++col)
              for (long row = 0; row < static_cast<long>(A_size1); ++row)
                wrapper_A(row, col) = wrapper_B(row, col) * data_alpha;
          }
        }
      }


      template <typename NumericT, typename F,
                typename ScalarType1, typename ScalarType2>
      void ambm(matrix_base<NumericT, F> & mat1,
                matrix_base<NumericT, F> const & mat2, ScalarType1 const & alpha, vcl_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
                matrix_base<NumericT, F> const & mat3, ScalarType2 const & beta,  vcl_size_t /*len_beta*/,  bool reciprocal_beta,  bool flip_sign_beta)
      {
        typedef NumericT        value_type;

        value_type       * data_A = detail::extract_raw_pointer<value_type>(mat1);
        value_type const * data_B = detail::extract_raw_pointer<value_type>(mat2);
        value_type const * data_C = detail::extract_raw_pointer<value_type>(mat3);

        value_type data_alpha = alpha;
        if (flip_sign_alpha)
          data_alpha = -data_alpha;

        value_type data_beta = beta;
        if (flip_sign_beta)
          data_beta = -data_beta;

        vcl_size_t A_start1 = viennacl::traits::start1(mat1);
        vcl_size_t A_start2 = viennacl::traits::start2(mat1);
        vcl_size_t A_inc1   = viennacl::traits::stride1(mat1);
        vcl_size_t A_inc2   = viennacl::traits::stride2(mat1);
        vcl_size_t A_size1  = viennacl::traits::size1(mat1);
        vcl_size_t A_size2  = viennacl::traits::size2(mat1);
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat1);
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat1);

        vcl_size_t B_start1 = viennacl::traits::start1(mat2);
        vcl_size_t B_start2 = viennacl::traits::start2(mat2);
        vcl_size_t B_inc1   = viennacl::traits::stride1(mat2);
        vcl_size_t B_inc2   = viennacl::traits::stride2(mat2);
        vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(mat2);
        vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(mat2);

        vcl_size_t C_start1 = viennacl::traits::start1(mat3);
        vcl_size_t C_start2 = viennacl::traits::start2(mat3);
        vcl_size_t C_inc1   = viennacl::traits::stride1(mat3);
        vcl_size_t C_inc2   = viennacl::traits::stride2(mat3);
        vcl_size_t C_internal_size1  = viennacl::traits::internal_size1(mat3);
        vcl_size_t C_internal_size2  = viennacl::traits::internal_size2(mat3);

        detail::matrix_array_wrapper<value_type,       typename F::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename F::orientation_category, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename F::orientation_category, false> wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

        if (detail::is_row_major(typename F::orientation_category()))
        {
          if (reciprocal_alpha && reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long row = 0; row < static_cast<long>(A_size1); ++row)
              for (long col = 0; col < static_cast<long>(A_size2); ++col)
                wrapper_A(row, col) = wrapper_B(row, col) / data_alpha + wrapper_C(row, col) / data_beta;
          }
          else if (reciprocal_alpha && !reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long row = 0; row < static_cast<long>(A_size1); ++row)
              for (long col = 0; col < static_cast<long>(A_size2); ++col)
                wrapper_A(row, col) = wrapper_B(row, col) / data_alpha + wrapper_C(row, col) * data_beta;
          }
          else if (!reciprocal_alpha && reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long row = 0; row < static_cast<long>(A_size1); ++row)
              for (long col = 0; col < static_cast<long>(A_size2); ++col)
                wrapper_A(row, col) = wrapper_B(row, col) * data_alpha + wrapper_C(row, col) / data_beta;
          }
          else if (!reciprocal_alpha && !reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long row = 0; row < static_cast<long>(A_size1); ++row)
              for (long col = 0; col < static_cast<long>(A_size2); ++col)
                wrapper_A(row, col) = wrapper_B(row, col) * data_alpha + wrapper_C(row, col) * data_beta;
          }
        }
        else
        {
          if (reciprocal_alpha && reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long col = 0; col < static_cast<long>(A_size2); ++col)
              for (long row = 0; row < static_cast<long>(A_size1); ++row)
                wrapper_A(row, col) = wrapper_B(row, col) / data_alpha + wrapper_C(row, col) / data_beta;
          }
          else if (reciprocal_alpha && !reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long col = 0; col < static_cast<long>(A_size2); ++col)
              for (long row = 0; row < static_cast<long>(A_size1); ++row)
                wrapper_A(row, col) = wrapper_B(row, col) / data_alpha + wrapper_C(row, col) * data_beta;
          }
          else if (!reciprocal_alpha && reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long col = 0; col < static_cast<long>(A_size2); ++col)
              for (long row = 0; row < static_cast<long>(A_size1); ++row)
                wrapper_A(row, col) = wrapper_B(row, col) * data_alpha + wrapper_C(row, col) / data_beta;
          }
          else if (!reciprocal_alpha && !reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long col = 0; col < static_cast<long>(A_size2); ++col)
              for (long row = 0; row < static_cast<long>(A_size1); ++row)
                wrapper_A(row, col) = wrapper_B(row, col) * data_alpha + wrapper_C(row, col) * data_beta;
          }
        }

      }


      template <typename NumericT, typename F,
                typename ScalarType1, typename ScalarType2>
      void ambm_m(matrix_base<NumericT, F> & mat1,
                  matrix_base<NumericT, F> const & mat2, ScalarType1 const & alpha, vcl_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
                  matrix_base<NumericT, F> const & mat3, ScalarType2 const & beta,  vcl_size_t /*len_beta*/,  bool reciprocal_beta,  bool flip_sign_beta)
      {
        typedef NumericT        value_type;

        value_type       * data_A = detail::extract_raw_pointer<value_type>(mat1);
        value_type const * data_B = detail::extract_raw_pointer<value_type>(mat2);
        value_type const * data_C = detail::extract_raw_pointer<value_type>(mat3);

        value_type data_alpha = alpha;
        if (flip_sign_alpha)
          data_alpha = -data_alpha;

        value_type data_beta = beta;
        if (flip_sign_beta)
          data_beta = -data_beta;

        vcl_size_t A_start1 = viennacl::traits::start1(mat1);
        vcl_size_t A_start2 = viennacl::traits::start2(mat1);
        vcl_size_t A_inc1   = viennacl::traits::stride1(mat1);
        vcl_size_t A_inc2   = viennacl::traits::stride2(mat1);
        vcl_size_t A_size1  = viennacl::traits::size1(mat1);
        vcl_size_t A_size2  = viennacl::traits::size2(mat1);
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat1);
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat1);

        vcl_size_t B_start1 = viennacl::traits::start1(mat2);
        vcl_size_t B_start2 = viennacl::traits::start2(mat2);
        vcl_size_t B_inc1   = viennacl::traits::stride1(mat2);
        vcl_size_t B_inc2   = viennacl::traits::stride2(mat2);
        vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(mat2);
        vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(mat2);

        vcl_size_t C_start1 = viennacl::traits::start1(mat3);
        vcl_size_t C_start2 = viennacl::traits::start2(mat3);
        vcl_size_t C_inc1   = viennacl::traits::stride1(mat3);
        vcl_size_t C_inc2   = viennacl::traits::stride2(mat3);
        vcl_size_t C_internal_size1  = viennacl::traits::internal_size1(mat3);
        vcl_size_t C_internal_size2  = viennacl::traits::internal_size2(mat3);

        detail::matrix_array_wrapper<value_type,       typename F::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename F::orientation_category, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename F::orientation_category, false> wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

        //typedef typename detail::majority_struct_for_orientation<typename M1::orientation_category>::type index_generator_A;
        //typedef typename detail::majority_struct_for_orientation<typename M2::orientation_category>::type index_generator_B;
        //typedef typename detail::majority_struct_for_orientation<typename M3::orientation_category>::type index_generator_C;

        if (detail::is_row_major(typename F::orientation_category()))
        {
          if (reciprocal_alpha && reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long row = 0; row < static_cast<long>(A_size1); ++row)
              for (long col = 0; col < static_cast<long>(A_size2); ++col)
                wrapper_A(row, col) += wrapper_B(row, col) / data_alpha + wrapper_C(row, col) / data_beta;
          }
          else if (reciprocal_alpha && !reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long row = 0; row < static_cast<long>(A_size1); ++row)
              for (long col = 0; col < static_cast<long>(A_size2); ++col)
                wrapper_A(row, col) += wrapper_B(row, col) / data_alpha + wrapper_C(row, col) * data_beta;
          }
          else if (!reciprocal_alpha && reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long row = 0; row < static_cast<long>(A_size1); ++row)
              for (long col = 0; col < static_cast<long>(A_size2); ++col)
                wrapper_A(row, col) += wrapper_B(row, col) * data_alpha + wrapper_C(row, col) / data_beta;
          }
          else if (!reciprocal_alpha && !reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long row = 0; row < static_cast<long>(A_size1); ++row)
              for (long col = 0; col < static_cast<long>(A_size2); ++col)
                wrapper_A(row, col) += wrapper_B(row, col) * data_alpha + wrapper_C(row, col) * data_beta;
          }
        }
        else
        {
          if (reciprocal_alpha && reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long col = 0; col < static_cast<long>(A_size2); ++col)
              for (long row = 0; row < static_cast<long>(A_size1); ++row)
                wrapper_A(row, col) += wrapper_B(row, col) / data_alpha + wrapper_C(row, col) / data_beta;
          }
          else if (reciprocal_alpha && !reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long col = 0; col < static_cast<long>(A_size2); ++col)
              for (long row = 0; row < static_cast<long>(A_size1); ++row)
                wrapper_A(row, col) += wrapper_B(row, col) / data_alpha + wrapper_C(row, col) * data_beta;
          }
          else if (!reciprocal_alpha && reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long col = 0; col < static_cast<long>(A_size2); ++col)
              for (long row = 0; row < static_cast<long>(A_size1); ++row)
                wrapper_A(row, col) += wrapper_B(row, col) * data_alpha + wrapper_C(row, col) / data_beta;
          }
          else if (!reciprocal_alpha && !reciprocal_beta)
          {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
            for (long col = 0; col < static_cast<long>(A_size2); ++col)
              for (long row = 0; row < static_cast<long>(A_size1); ++row)
                wrapper_A(row, col) += wrapper_B(row, col) * data_alpha + wrapper_C(row, col) * data_beta;
          }
        }

      }




      template <typename NumericT, typename F>
      void matrix_assign(matrix_base<NumericT, F> & mat, NumericT s, bool clear = false)
      {
        typedef NumericT        value_type;

        value_type       * data_A = detail::extract_raw_pointer<value_type>(mat);
        value_type alpha = static_cast<value_type>(s);

        vcl_size_t A_start1 = viennacl::traits::start1(mat);
        vcl_size_t A_start2 = viennacl::traits::start2(mat);
        vcl_size_t A_inc1   = viennacl::traits::stride1(mat);
        vcl_size_t A_inc2   = viennacl::traits::stride2(mat);
        vcl_size_t A_size1  = clear ? viennacl::traits::internal_size1(mat) : viennacl::traits::size1(mat);
        vcl_size_t A_size2  = clear ? viennacl::traits::internal_size2(mat) : viennacl::traits::size2(mat);
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat);

        detail::matrix_array_wrapper<value_type,       typename F::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

        if (detail::is_row_major(typename F::orientation_category()))
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (long row = 0; row < static_cast<long>(A_size1); ++row)
            for (long col = 0; col < static_cast<long>(A_size2); ++col)
              wrapper_A(row, col) = alpha;
              //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)]
              // = data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha;
        }
        else
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (long col = 0; col < static_cast<long>(A_size2); ++col)
            for (long row = 0; row < static_cast<long>(A_size1); ++row)
              wrapper_A(row, col) = alpha;
              //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)]
              // = data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha;
        }
      }



      template <typename NumericT, typename F>
      void matrix_diagonal_assign(matrix_base<NumericT, F> & mat, NumericT s)
      {
        typedef NumericT        value_type;

        value_type       * data_A = detail::extract_raw_pointer<value_type>(mat);
        value_type alpha = static_cast<value_type>(s);

        vcl_size_t A_start1 = viennacl::traits::start1(mat);
        vcl_size_t A_start2 = viennacl::traits::start2(mat);
        vcl_size_t A_inc1   = viennacl::traits::stride1(mat);
        vcl_size_t A_inc2   = viennacl::traits::stride2(mat);
        vcl_size_t A_size1  = viennacl::traits::size1(mat);
        //vcl_size_t A_size2  = viennacl::traits::size2(mat);
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat);

        detail::matrix_array_wrapper<value_type, typename F::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

#ifdef VIENNACL_WITH_OPENMP
        #pragma omp parallel for
#endif
        for (long row = 0; row < static_cast<long>(A_size1); ++row)
          wrapper_A(row, row) = alpha;
      }

      template <typename NumericT, typename F>
      void matrix_diag_from_vector(const vector_base<NumericT> & vec, int k, matrix_base<NumericT, F> & mat)
      {
        typedef NumericT        value_type;

        value_type       *data_A   = detail::extract_raw_pointer<value_type>(mat);
        value_type const *data_vec = detail::extract_raw_pointer<value_type>(vec);

        vcl_size_t A_start1 = viennacl::traits::start1(mat);
        vcl_size_t A_start2 = viennacl::traits::start2(mat);
        vcl_size_t A_inc1   = viennacl::traits::stride1(mat);
        vcl_size_t A_inc2   = viennacl::traits::stride2(mat);
        //vcl_size_t A_size1  = viennacl::traits::size1(mat);
        //vcl_size_t A_size2  = viennacl::traits::size2(mat);
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat);

        vcl_size_t v_start = viennacl::traits::start(vec);
        vcl_size_t v_inc   = viennacl::traits::stride(vec);
        vcl_size_t v_size  = viennacl::traits::size(vec);

        detail::matrix_array_wrapper<value_type, typename F::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

        vcl_size_t row_start = 0;
        vcl_size_t col_start = 0;

        if (k >= 0)
          col_start = static_cast<vcl_size_t>(k);
        else
          row_start = static_cast<vcl_size_t>(-k);

        matrix_assign(mat, NumericT(0));

        for (vcl_size_t i = 0; i < v_size; ++i)
          wrapper_A(row_start + i, col_start + i) = data_vec[v_start + i * v_inc];

      }

      template <typename NumericT, typename F>
      void matrix_diag_to_vector(const matrix_base<NumericT, F> & mat, int k, vector_base<NumericT> & vec)
      {
        typedef NumericT        value_type;

        value_type const *data_A   = detail::extract_raw_pointer<value_type>(mat);
        value_type       *data_vec = detail::extract_raw_pointer<value_type>(vec);

        vcl_size_t A_start1 = viennacl::traits::start1(mat);
        vcl_size_t A_start2 = viennacl::traits::start2(mat);
        vcl_size_t A_inc1   = viennacl::traits::stride1(mat);
        vcl_size_t A_inc2   = viennacl::traits::stride2(mat);
        //vcl_size_t A_size1  = viennacl::traits::size1(mat);
        //vcl_size_t A_size2  = viennacl::traits::size2(mat);
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat);

        vcl_size_t v_start = viennacl::traits::start(vec);
        vcl_size_t v_inc   = viennacl::traits::stride(vec);
        vcl_size_t v_size  = viennacl::traits::size(vec);

        detail::matrix_array_wrapper<value_type const, typename F::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

        vcl_size_t row_start = 0;
        vcl_size_t col_start = 0;

        if (k >= 0)
          col_start = static_cast<vcl_size_t>(k);
        else
          row_start = static_cast<vcl_size_t>(-k);

        for (vcl_size_t i = 0; i < v_size; ++i)
          data_vec[v_start + i * v_inc] = wrapper_A(row_start + i, col_start + i);
      }

      template <typename NumericT, typename F>
      void matrix_row(const matrix_base<NumericT, F> & mat, unsigned int i, vector_base<NumericT> & vec)
      {
        typedef NumericT        value_type;

        value_type const *data_A   = detail::extract_raw_pointer<value_type>(mat);
        value_type       *data_vec = detail::extract_raw_pointer<value_type>(vec);

        vcl_size_t A_start1 = viennacl::traits::start1(mat);
        vcl_size_t A_start2 = viennacl::traits::start2(mat);
        vcl_size_t A_inc1   = viennacl::traits::stride1(mat);
        vcl_size_t A_inc2   = viennacl::traits::stride2(mat);
        //vcl_size_t A_size1  = viennacl::traits::size1(mat);
        //vcl_size_t A_size2  = viennacl::traits::size2(mat);
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat);

        vcl_size_t v_start = viennacl::traits::start(vec);
        vcl_size_t v_inc   = viennacl::traits::stride(vec);
        vcl_size_t v_size  = viennacl::traits::size(vec);

        detail::matrix_array_wrapper<value_type const, typename F::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

        for (vcl_size_t j = 0; j < v_size; ++j)
          data_vec[v_start + j * v_inc] = wrapper_A(i, j);
      }

      template <typename NumericT, typename F>
      void matrix_column(const matrix_base<NumericT, F> & mat, unsigned int j, vector_base<NumericT> & vec)
      {
        typedef NumericT        value_type;

        value_type const *data_A   = detail::extract_raw_pointer<value_type>(mat);
        value_type       *data_vec = detail::extract_raw_pointer<value_type>(vec);

        vcl_size_t A_start1 = viennacl::traits::start1(mat);
        vcl_size_t A_start2 = viennacl::traits::start2(mat);
        vcl_size_t A_inc1   = viennacl::traits::stride1(mat);
        vcl_size_t A_inc2   = viennacl::traits::stride2(mat);
        //vcl_size_t A_size1  = viennacl::traits::size1(mat);
        //vcl_size_t A_size2  = viennacl::traits::size2(mat);
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat);

        vcl_size_t v_start = viennacl::traits::start(vec);
        vcl_size_t v_inc   = viennacl::traits::stride(vec);
        vcl_size_t v_size  = viennacl::traits::size(vec);

        detail::matrix_array_wrapper<value_type const, typename F::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

        for (vcl_size_t i = 0; i < v_size; ++i)
          data_vec[v_start + i * v_inc] = wrapper_A(i, j);
      }

      //
      ///////////////////////// Element-wise operation //////////////////////////////////
      //

      // Binary operations A = B .* C and A = B ./ C

      /** @brief Implementation of the element-wise operations A = B .* C and A = B ./ C    (using MATLAB syntax)
      *
      * @param A      The result matrix (or -range, or -slice)
      * @param proxy  The proxy object holding B, C, and the operation
      */
      template <typename NumericT, typename F, typename OP>
      void element_op(matrix_base<NumericT, F> & A,
                      matrix_expression<const matrix_base<NumericT, F>, const matrix_base<NumericT, F>, op_element_binary<OP> > const & proxy)
      {
        typedef NumericT        value_type;
        typedef viennacl::linalg::detail::op_applier<op_element_binary<OP> >    OpFunctor;

        value_type       * data_A = detail::extract_raw_pointer<value_type>(A);
        value_type const * data_B = detail::extract_raw_pointer<value_type>(proxy.lhs());
        value_type const * data_C = detail::extract_raw_pointer<value_type>(proxy.rhs());

        vcl_size_t A_start1 = viennacl::traits::start1(A);
        vcl_size_t A_start2 = viennacl::traits::start2(A);
        vcl_size_t A_inc1   = viennacl::traits::stride1(A);
        vcl_size_t A_inc2   = viennacl::traits::stride2(A);
        vcl_size_t A_size1  = viennacl::traits::size1(A);
        vcl_size_t A_size2  = viennacl::traits::size2(A);
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(A);
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(A);

        vcl_size_t B_start1 = viennacl::traits::start1(proxy.lhs());
        vcl_size_t B_start2 = viennacl::traits::start2(proxy.lhs());
        vcl_size_t B_inc1   = viennacl::traits::stride1(proxy.lhs());
        vcl_size_t B_inc2   = viennacl::traits::stride2(proxy.lhs());
        vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(proxy.lhs());
        vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(proxy.lhs());

        vcl_size_t C_start1 = viennacl::traits::start1(proxy.rhs());
        vcl_size_t C_start2 = viennacl::traits::start2(proxy.rhs());
        vcl_size_t C_inc1   = viennacl::traits::stride1(proxy.rhs());
        vcl_size_t C_inc2   = viennacl::traits::stride2(proxy.rhs());
        vcl_size_t C_internal_size1  = viennacl::traits::internal_size1(proxy.rhs());
        vcl_size_t C_internal_size2  = viennacl::traits::internal_size2(proxy.rhs());

        detail::matrix_array_wrapper<value_type,       typename F::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename F::orientation_category, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename F::orientation_category, false> wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

        if (detail::is_row_major(typename F::orientation_category()))
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (long row = 0; row < static_cast<long>(A_size1); ++row)
            for (long col = 0; col < static_cast<long>(A_size2); ++col)
              OpFunctor::apply(wrapper_A(row, col), wrapper_B(row, col), wrapper_C(row, col));
              //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)]
              // =   data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha
              //   + data_C[index_generator_C::mem_index(row * C_inc1 + C_start1, col * C_inc2 + C_start2, C_internal_size1, C_internal_size2)] * beta;
        }
        else
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (long col = 0; col < static_cast<long>(A_size2); ++col)
            for (long row = 0; row < static_cast<long>(A_size1); ++row)
              OpFunctor::apply(wrapper_A(row, col), wrapper_B(row, col), wrapper_C(row, col));

              //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)]
              // =   data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha
              //   + data_C[index_generator_C::mem_index(row * C_inc1 + C_start1, col * C_inc2 + C_start2, C_internal_size1, C_internal_size2)] * beta;
        }
      }

      // Unary operations

      // A = op(B)
      template <typename NumericT, typename F, typename OP>
      void element_op(matrix_base<NumericT, F> & A,
                      matrix_expression<const matrix_base<NumericT, F>, const matrix_base<NumericT, F>, op_element_unary<OP> > const & proxy)
      {
        typedef NumericT        value_type;
        typedef viennacl::linalg::detail::op_applier<op_element_unary<OP> >    OpFunctor;

        value_type       * data_A = detail::extract_raw_pointer<value_type>(A);
        value_type const * data_B = detail::extract_raw_pointer<value_type>(proxy.lhs());

        vcl_size_t A_start1 = viennacl::traits::start1(A);
        vcl_size_t A_start2 = viennacl::traits::start2(A);
        vcl_size_t A_inc1   = viennacl::traits::stride1(A);
        vcl_size_t A_inc2   = viennacl::traits::stride2(A);
        vcl_size_t A_size1  = viennacl::traits::size1(A);
        vcl_size_t A_size2  = viennacl::traits::size2(A);
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(A);
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(A);

        vcl_size_t B_start1 = viennacl::traits::start1(proxy.lhs());
        vcl_size_t B_start2 = viennacl::traits::start2(proxy.lhs());
        vcl_size_t B_inc1   = viennacl::traits::stride1(proxy.lhs());
        vcl_size_t B_inc2   = viennacl::traits::stride2(proxy.lhs());
        vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(proxy.lhs());
        vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(proxy.lhs());

        detail::matrix_array_wrapper<value_type,       typename F::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename F::orientation_category, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);

        if (detail::is_row_major(typename F::orientation_category()))
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (long row = 0; row < static_cast<long>(A_size1); ++row)
            for (long col = 0; col < static_cast<long>(A_size2); ++col)
              OpFunctor::apply(wrapper_A(row, col), wrapper_B(row, col));
        }
        else
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (long col = 0; col < static_cast<long>(A_size2); ++col)
            for (long row = 0; row < static_cast<long>(A_size1); ++row)
              OpFunctor::apply(wrapper_A(row, col), wrapper_B(row, col));
        }
      }



      //
      /////////////////////////   matrix-vector products /////////////////////////////////
      //

      // A * x

      /** @brief Carries out matrix-vector multiplication
      *
      * Implementation of the convenience expression result = prod(mat, vec);
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
      template <typename NumericT, typename F>
      void prod_impl(const matrix_base<NumericT, F> & mat,
                     const vector_base<NumericT> & vec,
                           vector_base<NumericT> & result)
      {
        typedef NumericT        value_type;

        value_type const * data_A = detail::extract_raw_pointer<value_type>(mat);
        value_type const * data_x = detail::extract_raw_pointer<value_type>(vec);
        value_type       * data_result = detail::extract_raw_pointer<value_type>(result);

        vcl_size_t A_start1 = viennacl::traits::start1(mat);
        vcl_size_t A_start2 = viennacl::traits::start2(mat);
        vcl_size_t A_inc1   = viennacl::traits::stride1(mat);
        vcl_size_t A_inc2   = viennacl::traits::stride2(mat);
        vcl_size_t A_size1  = viennacl::traits::size1(mat);
        vcl_size_t A_size2  = viennacl::traits::size2(mat);
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat);

        vcl_size_t start1 = viennacl::traits::start(vec);
        vcl_size_t inc1   = viennacl::traits::stride(vec);

        vcl_size_t start2 = viennacl::traits::start(result);
        vcl_size_t inc2   = viennacl::traits::stride(result);

        if (detail::is_row_major(typename F::orientation_category()))
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (long row = 0; row < static_cast<long>(A_size1); ++row)
          {
            value_type temp = 0;
            for (vcl_size_t col = 0; col < A_size2; ++col)
              temp += data_A[viennacl::row_major::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] * data_x[col * inc1 + start1];

            data_result[row * inc2 + start2] = temp;
          }
        }
        else
        {
          {
            value_type temp = data_x[start1];
            for (vcl_size_t row = 0; row < A_size1; ++row)
              data_result[row * inc2 + start2] = data_A[viennacl::column_major::mem_index(row * A_inc1 + A_start1, A_start2, A_internal_size1, A_internal_size2)] * temp;
          }
          for (vcl_size_t col = 1; col < A_size2; ++col)  //run through matrix sequentially
          {
            value_type temp = data_x[col * inc1 + start1];
            for (vcl_size_t row = 0; row < A_size1; ++row)
              data_result[row * inc2 + start2] += data_A[viennacl::column_major::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] * temp;
          }
        }
      }


      // trans(A) * x

      /** @brief Carries out matrix-vector multiplication with a transposed matrix
      *
      * Implementation of the convenience expression result = trans(mat) * vec;
      *
      * @param mat_trans  The transposed matrix proxy
      * @param vec        The vector
      * @param result     The result vector
      */
      template <typename NumericT, typename F>
      void prod_impl(const viennacl::matrix_expression< const matrix_base<NumericT, F>, const matrix_base<NumericT, F>, op_trans> & mat_trans,
                     const vector_base<NumericT> & vec,
                           vector_base<NumericT> & result)
      {
        typedef NumericT        value_type;

        value_type const * data_A = detail::extract_raw_pointer<value_type>(mat_trans.lhs());
        value_type const * data_x = detail::extract_raw_pointer<value_type>(vec);
        value_type       * data_result = detail::extract_raw_pointer<value_type>(result);

        vcl_size_t A_start1 = viennacl::traits::start1(mat_trans.lhs());
        vcl_size_t A_start2 = viennacl::traits::start2(mat_trans.lhs());
        vcl_size_t A_inc1   = viennacl::traits::stride1(mat_trans.lhs());
        vcl_size_t A_inc2   = viennacl::traits::stride2(mat_trans.lhs());
        vcl_size_t A_size1  = viennacl::traits::size1(mat_trans.lhs());
        vcl_size_t A_size2  = viennacl::traits::size2(mat_trans.lhs());
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat_trans.lhs());
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat_trans.lhs());

        vcl_size_t start1 = viennacl::traits::start(vec);
        vcl_size_t inc1   = viennacl::traits::stride(vec);

        vcl_size_t start2 = viennacl::traits::start(result);
        vcl_size_t inc2   = viennacl::traits::stride(result);

        if (detail::is_row_major(typename F::orientation_category()))
        {
          {
            value_type temp = data_x[start1];
            for (vcl_size_t row = 0; row < A_size2; ++row)
              data_result[row * inc2 + start2] = data_A[viennacl::row_major::mem_index(A_start1, row * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] * temp;
          }

          for (vcl_size_t col = 1; col < A_size1; ++col)  //run through matrix sequentially
          {
            value_type temp = data_x[col * inc1 + start1];
            for (vcl_size_t row = 0; row < A_size2; ++row)
            {
              data_result[row * inc2 + start2] += data_A[viennacl::row_major::mem_index(col * A_inc1 + A_start1, row * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] * temp;
            }
          }
        }
        else
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (long row = 0; row < static_cast<long>(A_size2); ++row)
          {
            value_type temp = 0;
            for (vcl_size_t col = 0; col < A_size1; ++col)
              temp += data_A[viennacl::column_major::mem_index(col * A_inc1 + A_start1, row * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] * data_x[col * inc1 + start1];

            data_result[row * inc2 + start2] = temp;
          }
        }
      }


      //
      /////////////////////////   matrix-matrix products /////////////////////////////////
      //

      namespace detail
      {
        template <typename A, typename B, typename C, typename NumericT>
        void prod(A & a, B & b, C & c,
                  vcl_size_t C_size1, vcl_size_t C_size2, vcl_size_t A_size2,
                  NumericT alpha, NumericT beta)
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (long i=0; i<static_cast<long>(C_size1); ++i)
          {
            for (vcl_size_t j=0; j<C_size2; ++j)
            {
              NumericT temp = 0;
              for (vcl_size_t k=0; k<A_size2; ++k)
                temp += a(i, k) * b(k, j);

              temp *= alpha;
              if (beta != 0)
                temp += beta * c(i,j);
              c(i,j) = temp;
            }
          }
        }

      }

      /** @brief Carries out matrix-matrix multiplication
      *
      * Implementation of C = prod(A, B);
      *
      */
      template <typename NumericT, typename F1, typename F2, typename F3, typename ScalarType >
      void prod_impl(const matrix_base<NumericT, F1> & A,
                     const matrix_base<NumericT, F2> & B,
                           matrix_base<NumericT, F3> & C,
                     ScalarType alpha,
                     ScalarType beta)
      {
        typedef NumericT        value_type;

        value_type const * data_A = detail::extract_raw_pointer<value_type>(A);
        value_type const * data_B = detail::extract_raw_pointer<value_type>(B);
        value_type       * data_C = detail::extract_raw_pointer<value_type>(C);

        vcl_size_t A_start1 = viennacl::traits::start1(A);
        vcl_size_t A_start2 = viennacl::traits::start2(A);
        vcl_size_t A_inc1   = viennacl::traits::stride1(A);
        vcl_size_t A_inc2   = viennacl::traits::stride2(A);
        vcl_size_t A_size2  = viennacl::traits::size2(A);
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(A);
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(A);

        vcl_size_t B_start1 = viennacl::traits::start1(B);
        vcl_size_t B_start2 = viennacl::traits::start2(B);
        vcl_size_t B_inc1   = viennacl::traits::stride1(B);
        vcl_size_t B_inc2   = viennacl::traits::stride2(B);
        vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(B);
        vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(B);

        vcl_size_t C_start1 = viennacl::traits::start1(C);
        vcl_size_t C_start2 = viennacl::traits::start2(C);
        vcl_size_t C_inc1   = viennacl::traits::stride1(C);
        vcl_size_t C_inc2   = viennacl::traits::stride2(C);
        vcl_size_t C_size1  = viennacl::traits::size1(C);
        vcl_size_t C_size2  = viennacl::traits::size2(C);
        vcl_size_t C_internal_size1  = viennacl::traits::internal_size1(C);
        vcl_size_t C_internal_size2  = viennacl::traits::internal_size2(C);

        detail::matrix_array_wrapper<value_type const, typename F1::orientation_category, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename F2::orientation_category, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        detail::matrix_array_wrapper<value_type,       typename F3::orientation_category, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

        detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
      }



      /** @brief Carries out matrix-matrix multiplication
      *
      * Implementation of C = prod(trans(A), B);
      *
      */
      template <typename NumericT, typename F1, typename F2, typename F3, typename ScalarType >
      void prod_impl(const viennacl::matrix_expression< const matrix_base<NumericT, F1>,
                                                        const matrix_base<NumericT, F1>,
                                                        op_trans> & A,
                     const matrix_base<NumericT, F2> & B,
                           matrix_base<NumericT, F3> & C,
                     ScalarType alpha,
                     ScalarType beta)
      {
        typedef NumericT        value_type;

        value_type const * data_A = detail::extract_raw_pointer<value_type>(A.lhs());
        value_type const * data_B = detail::extract_raw_pointer<value_type>(B);
        value_type       * data_C = detail::extract_raw_pointer<value_type>(C);

        vcl_size_t A_start1 = viennacl::traits::start1(A.lhs());
        vcl_size_t A_start2 = viennacl::traits::start2(A.lhs());
        vcl_size_t A_inc1   = viennacl::traits::stride1(A.lhs());
        vcl_size_t A_inc2   = viennacl::traits::stride2(A.lhs());
        vcl_size_t A_size1  = viennacl::traits::size1(A.lhs());
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(A.lhs());
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(A.lhs());

        vcl_size_t B_start1 = viennacl::traits::start1(B);
        vcl_size_t B_start2 = viennacl::traits::start2(B);
        vcl_size_t B_inc1   = viennacl::traits::stride1(B);
        vcl_size_t B_inc2   = viennacl::traits::stride2(B);
        vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(B);
        vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(B);

        vcl_size_t C_start1 = viennacl::traits::start1(C);
        vcl_size_t C_start2 = viennacl::traits::start2(C);
        vcl_size_t C_inc1   = viennacl::traits::stride1(C);
        vcl_size_t C_inc2   = viennacl::traits::stride2(C);
        vcl_size_t C_size1  = viennacl::traits::size1(C);
        vcl_size_t C_size2  = viennacl::traits::size2(C);
        vcl_size_t C_internal_size1  = viennacl::traits::internal_size1(C);
        vcl_size_t C_internal_size2  = viennacl::traits::internal_size2(C);

        detail::matrix_array_wrapper<value_type const, typename F1::orientation_category, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename F2::orientation_category, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        detail::matrix_array_wrapper<value_type,       typename F3::orientation_category, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

        detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
      }




      /** @brief Carries out matrix-matrix multiplication
      *
      * Implementation of C = prod(A, trans(B));
      *
      */
      template <typename NumericT, typename F1, typename F2, typename F3, typename ScalarType >
      void prod_impl(const matrix_base<NumericT, F1> & A,
                     const viennacl::matrix_expression< const matrix_base<NumericT, F2>, const matrix_base<NumericT, F2>, op_trans> & B,
                           matrix_base<NumericT, F3> & C,
                     ScalarType alpha,
                     ScalarType beta)
      {
        typedef NumericT        value_type;

        value_type const * data_A = detail::extract_raw_pointer<value_type>(A);
        value_type const * data_B = detail::extract_raw_pointer<value_type>(B.lhs());
        value_type       * data_C = detail::extract_raw_pointer<value_type>(C);

        vcl_size_t A_start1 = viennacl::traits::start1(A);
        vcl_size_t A_start2 = viennacl::traits::start2(A);
        vcl_size_t A_inc1   = viennacl::traits::stride1(A);
        vcl_size_t A_inc2   = viennacl::traits::stride2(A);
        vcl_size_t A_size2  = viennacl::traits::size2(A);
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(A);
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(A);

        vcl_size_t B_start1 = viennacl::traits::start1(B.lhs());
        vcl_size_t B_start2 = viennacl::traits::start2(B.lhs());
        vcl_size_t B_inc1   = viennacl::traits::stride1(B.lhs());
        vcl_size_t B_inc2   = viennacl::traits::stride2(B.lhs());
        vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(B.lhs());
        vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(B.lhs());

        vcl_size_t C_start1 = viennacl::traits::start1(C);
        vcl_size_t C_start2 = viennacl::traits::start2(C);
        vcl_size_t C_inc1   = viennacl::traits::stride1(C);
        vcl_size_t C_inc2   = viennacl::traits::stride2(C);
        vcl_size_t C_size1  = viennacl::traits::size1(C);
        vcl_size_t C_size2  = viennacl::traits::size2(C);
        vcl_size_t C_internal_size1  = viennacl::traits::internal_size1(C);
        vcl_size_t C_internal_size2  = viennacl::traits::internal_size2(C);

        detail::matrix_array_wrapper<value_type const, typename F1::orientation_category, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename F2::orientation_category, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        detail::matrix_array_wrapper<value_type,       typename F3::orientation_category, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

        detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
      }



      /** @brief Carries out matrix-matrix multiplication
      *
      * Implementation of C = prod(trans(A), trans(B));
      *
      */
      template <typename NumericT, typename F1, typename F2, typename F3, typename ScalarType >
      void prod_impl(const viennacl::matrix_expression< const matrix_base<NumericT, F1>, const matrix_base<NumericT, F1>, op_trans> & A,
                     const viennacl::matrix_expression< const matrix_base<NumericT, F2>, const matrix_base<NumericT, F2>, op_trans> & B,
                     matrix_base<NumericT, F3> & C,
                     ScalarType alpha,
                     ScalarType beta)
      {
        typedef NumericT        value_type;

        value_type const * data_A = detail::extract_raw_pointer<value_type>(A.lhs());
        value_type const * data_B = detail::extract_raw_pointer<value_type>(B.lhs());
        value_type       * data_C = detail::extract_raw_pointer<value_type>(C);

        vcl_size_t A_start1 = viennacl::traits::start1(A.lhs());
        vcl_size_t A_start2 = viennacl::traits::start2(A.lhs());
        vcl_size_t A_inc1   = viennacl::traits::stride1(A.lhs());
        vcl_size_t A_inc2   = viennacl::traits::stride2(A.lhs());
        vcl_size_t A_size1  = viennacl::traits::size1(A.lhs());
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(A.lhs());
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(A.lhs());

        vcl_size_t B_start1 = viennacl::traits::start1(B.lhs());
        vcl_size_t B_start2 = viennacl::traits::start2(B.lhs());
        vcl_size_t B_inc1   = viennacl::traits::stride1(B.lhs());
        vcl_size_t B_inc2   = viennacl::traits::stride2(B.lhs());
        vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(B.lhs());
        vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(B.lhs());

        vcl_size_t C_start1 = viennacl::traits::start1(C);
        vcl_size_t C_start2 = viennacl::traits::start2(C);
        vcl_size_t C_inc1   = viennacl::traits::stride1(C);
        vcl_size_t C_inc2   = viennacl::traits::stride2(C);
        vcl_size_t C_size1  = viennacl::traits::size1(C);
        vcl_size_t C_size2  = viennacl::traits::size2(C);
        vcl_size_t C_internal_size1  = viennacl::traits::internal_size1(C);
        vcl_size_t C_internal_size2  = viennacl::traits::internal_size2(C);

        detail::matrix_array_wrapper<value_type const, typename F1::orientation_category, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename F2::orientation_category, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        detail::matrix_array_wrapper<value_type,       typename F3::orientation_category, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

        detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
      }




      //
      /////////////////////////   miscellaneous operations /////////////////////////////////
      //


      /** @brief The implementation of the operation mat += alpha * vec1 * vec2^T, i.e. a scaled rank 1 update
      *
      * Implementation of the convenience expression result += alpha * outer_prod(vec1, vec2);
      *
      * @param mat1    The matrix to be updated
      * @param alpha            The scaling factor (either a viennacl::scalar<>, float, or double)
      * @param reciprocal_alpha Use 1/alpha instead of alpha
      * @param flip_sign_alpha  Use -alpha instead of alpha
      * @param vec1    The first vector
      * @param vec2    The second vector
      */
      template <typename NumericT, typename F, typename S1>
      void scaled_rank_1_update(matrix_base<NumericT, F> & mat1,
                                S1 const & alpha, vcl_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
                                const vector_base<NumericT> & vec1,
                                const vector_base<NumericT> & vec2)
      {
        typedef NumericT        value_type;

        value_type       * data_A  = detail::extract_raw_pointer<value_type>(mat1);
        value_type const * data_v1 = detail::extract_raw_pointer<value_type>(vec1);
        value_type const * data_v2 = detail::extract_raw_pointer<value_type>(vec2);

        vcl_size_t A_start1 = viennacl::traits::start1(mat1);
        vcl_size_t A_start2 = viennacl::traits::start2(mat1);
        vcl_size_t A_inc1   = viennacl::traits::stride1(mat1);
        vcl_size_t A_inc2   = viennacl::traits::stride2(mat1);
        vcl_size_t A_size1  = viennacl::traits::size1(mat1);
        vcl_size_t A_size2  = viennacl::traits::size2(mat1);
        vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat1);
        vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat1);

        vcl_size_t start1 = viennacl::traits::start(vec1);
        vcl_size_t inc1   = viennacl::traits::stride(vec1);

        vcl_size_t start2 = viennacl::traits::start(vec2);
        vcl_size_t inc2   = viennacl::traits::stride(vec2);

        value_type data_alpha = alpha;
        if (flip_sign_alpha)
          data_alpha = -data_alpha;
        if (reciprocal_alpha)
          data_alpha = static_cast<value_type>(1) / data_alpha;

        if (detail::is_row_major(typename F::orientation_category()))
        {
          for (vcl_size_t row = 0; row < A_size1; ++row)
          {
            value_type value_v1 = data_alpha * data_v1[row * inc1 + start1];
            for (vcl_size_t col = 0; col < A_size2; ++col)
              data_A[viennacl::row_major::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] += value_v1 * data_v2[col * inc2 + start2];
          }
        }
        else
        {
          for (vcl_size_t col = 0; col < A_size2; ++col)  //run through matrix sequentially
          {
            value_type value_v2 = data_alpha * data_v2[col * inc2 + start2];
            for (vcl_size_t row = 0; row < A_size1; ++row)
              data_A[viennacl::column_major::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] += data_v1[row * inc1 + start1] * value_v2;
          }
        }
      }

      template <typename NumericT, typename F, typename S1>
       void bidiag_pack_kernel(matrix_base<NumericT, F> & A,
                        vector_base<S1> & D,
                        vector_base<S1> & S
                        )

       {
         //std::cout << "Bidiag_pack host based!!\n";

         typedef NumericT        value_type;

         value_type * data_A  = detail::extract_raw_pointer<value_type>(A);
         value_type * data_D = detail::extract_raw_pointer<value_type>(D);
         value_type * data_S = detail::extract_raw_pointer<value_type>(S);

         vcl_size_t A_start1 = viennacl::traits::start1(A);
         vcl_size_t A_start2 = viennacl::traits::start2(A);
         vcl_size_t A_inc1   = viennacl::traits::stride1(A);
         vcl_size_t A_inc2   = viennacl::traits::stride2(A);
         vcl_size_t A_size1  = viennacl::traits::size1(A);
         vcl_size_t A_size2  = viennacl::traits::size2(A);
         vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(A);
         vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(A);

         vcl_size_t start1 = viennacl::traits::start(D);
         vcl_size_t inc1   = viennacl::traits::stride(D);
         vcl_size_t size1  = viennacl::traits::size(D);

         vcl_size_t start2 = viennacl::traits::start(S);
         vcl_size_t inc2   = viennacl::traits::stride(S);
         vcl_size_t size2  = viennacl::traits::size(S);

         vcl_size_t size = std::min(size1, size2);
         if (detail::is_row_major(typename F::orientation_category()))
         {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
             for(vcl_size_t i = 0;  i < size -1; i++)
               {

                 data_D[start1 + inc1 * i] =        data_A[viennacl::row_major::mem_index(i * A_inc1 + A_start1, i * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
                 data_S[start2 + inc2 * (i + 1)] =  data_A[viennacl::row_major::mem_index(i * A_inc1 + A_start1, (i + 1) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
               }
             data_D[start1 + inc1 * (size-1)] = data_A[viennacl::row_major::mem_index((size-1) * A_inc1 + A_start1, (size-1) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];

          }
         else
           {
#ifdef VIENNACL_WITH_OPENMP
            #pragma omp parallel for
#endif
             for(vcl_size_t i = 0;  i < size -1; i++)
               {
                 data_D[start1 + inc1 * i] =        data_A[viennacl::column_major::mem_index(i * A_inc1 + A_start1, i * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
                 data_S[start2 + inc2 * (i + 1)] =  data_A[viennacl::column_major::mem_index(i * A_inc1 + A_start1, (i + 1) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
               }
             data_D[start1 + inc1 * (size-1)] = data_A[viennacl::column_major::mem_index((size-1) * A_inc1 + A_start1, (size-1) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
           }

       }



       template <typename NumericT, typename F, typename VectorType>
        void bidiag_pack(matrix_base<NumericT, F> & A,
                         VectorType & dh,
                         VectorType & sh
                        )
        {
          std::cout << "bidiag_pack!\n";
          viennacl::vector<NumericT> D(dh.size());
          viennacl::vector<NumericT> S(sh.size());
          viennacl::copy(dh, D);
          viennacl::copy(sh, S);

          viennacl::linalg::host_based::bidiag_pack_kernel(A, D, S);

          viennacl::copy(D, dh);
          viennacl::copy(S, sh);

        }



       template <typename NumericT, typename F>
       void house_update_A_left(matrix_base<NumericT, F>& A,
                                vector_base<NumericT> & D,
                                vcl_size_t start)
       {         
         //std::cout << "house_update_A_left host based started!!\n";
         NumericT temp = 0;
         NumericT beta = 0;

         viennacl::vector<NumericT> vcl_w(D.size());
         viennacl::vector<NumericT> vcl_D(D.size());
         viennacl::copy(D, vcl_D);

         temp = inner_prod(vcl_D, vcl_D);
         beta = 2/temp;

         vcl_w = beta*viennacl::linalg::prod(trans(A), vcl_D);
         viennacl::linalg::host_based::scaled_rank_1_update(A, 1, 1, 0, 1, vcl_D, vcl_w);
       }

       template <typename NumericT, typename F>
       void house_update_A_right(matrix_base<NumericT, F>& A,
                                 vector_base<NumericT> & D,
                                 vcl_size_t start)
       {
         //std::cout << "house_update_A_right host based started!!\n";

         NumericT temp = 0;
         NumericT beta = 0;

         viennacl::vector<NumericT> vcl_w(D.size());
         viennacl::vector<NumericT> vcl_D(D.size());
         viennacl::copy(D, vcl_D);

         temp = inner_prod(vcl_D, vcl_D);
         beta = 2/temp;

         vcl_w = beta*viennacl::linalg::prod(A, vcl_D);
         viennacl::linalg::host_based::scaled_rank_1_update(A, 1, 1, 0, 1, vcl_w, vcl_D);
       }


       template <typename NumericT, typename F>
       void house_update_QL(matrix_base<NumericT, F>& A,
                            matrix_base<NumericT, F>& Q,
                            vector_base<NumericT> & D)

       {
         //std::cout << "house_update_QL host based started!!\n";
         NumericT temp, beta = 0;
         viennacl::matrix<NumericT> vcl_P(A.size1(), A.size2());

         viennacl::matrix<NumericT> I(A.size1(), A.size2());
         viennacl::matrix<NumericT> Q_temp(Q.size1(), Q.size2());
         copy(Q, Q_temp);
         std::cout << D << std::endl;
         vcl_P = viennacl::identity_matrix<NumericT>(A.size1());
         I     = viennacl::identity_matrix<NumericT>(A.size1());
         viennacl::vector<NumericT> vcl_D(D.size());
         viennacl::copy(D, vcl_D);

         temp = inner_prod(vcl_D, vcl_D);
         beta = 2/temp;
         std::cout << beta << std::endl;

         viennacl::linalg::host_based::scaled_rank_1_update(vcl_P, beta, 1, 0, 1, vcl_D, vcl_D);  //scaled_rank_1_update in linalg/matrix_operations.hpp beschrieben
         //std::cout << "\n\nMatrix Q_temp:\n" << Q_temp << std::endl;
         Q = prod(Q_temp, vcl_P);  //P wurde korrekt berechnet - ueberprueft

       }

       template<typename NumericT, typename F>
         void givens_next(matrix_base<NumericT, F>& Q,
                         vector_base<NumericT> & tmp1,    //cs
                         vector_base<NumericT> & tmp2,    //ss
                         int l,                           //start_i
                         int m                            //end_i+1
                       )
         {
           typedef NumericT        value_type;

           value_type * data_Q  = detail::extract_raw_pointer<value_type>(Q);
           value_type * data_tmp1 = detail::extract_raw_pointer<value_type>(tmp1);
           value_type * data_tmp2 = detail::extract_raw_pointer<value_type>(tmp2);

           vcl_size_t Q_start1 = viennacl::traits::start1(Q);
           vcl_size_t Q_start2 = viennacl::traits::start2(Q);
           vcl_size_t Q_inc1   = viennacl::traits::stride1(Q);
           vcl_size_t Q_inc2   = viennacl::traits::stride2(Q);
           vcl_size_t Q_size1  = viennacl::traits::size1(Q);
           vcl_size_t Q_size2  = viennacl::traits::size2(Q);
           vcl_size_t Q_internal_size1  = viennacl::traits::internal_size1(Q);
           vcl_size_t Q_internal_size2  = viennacl::traits::internal_size2(Q);

           vcl_size_t start1 = viennacl::traits::start(tmp1);
           vcl_size_t inc1   = viennacl::traits::stride(tmp1);
           vcl_size_t size1  = viennacl::traits::size(tmp1);

           vcl_size_t start2 = viennacl::traits::start(tmp2);
           vcl_size_t inc2   = viennacl::traits::stride(tmp2);
           vcl_size_t size2  = viennacl::traits::size(tmp2);



           //std::cout << "givens_next host based started!\n";if (detail::is_row_major(typename F::orientation_category()))



           if (detail::is_row_major(typename F::orientation_category()))
           {
               for( int i = m - 1; i >= l; i--)
                 {
                   for(uint k = 0; k < Q_size1; k++)
                     {

                      // h = data_Q(i+1, k);
                       NumericT h = data_Q[viennacl::row_major::mem_index((i + 1) * Q_inc1 + Q_start1, k * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)];

                       //Q(i+1, k) = tmp2[i] * Q(i, k) + tmp1[i]*h;
                       data_Q[viennacl::row_major::mem_index((i + 1) * Q_inc1 + Q_start1, k * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] = tmp2[start2 + inc2 * i] *
                           data_Q[viennacl::row_major::mem_index(i  * Q_inc1 + Q_start1, k * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] + tmp1[start1 + inc1 * i] * h;

                       //Q(i,   k) = tmp1[i] * Q(i, k) - tmp2[i]*h;
                       data_Q[viennacl::row_major::mem_index(i  * Q_inc1 + Q_start1, k * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] = tmp1[start1 + inc1 * i] *
                           data_Q[viennacl::row_major::mem_index(i  * Q_inc1 + Q_start1, k * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] - tmp2[start2 + inc2 * i]*h;
                     }
                 }
           }
           else       // column_major
             {
               for( int i = m - 1; i >= l; i--)
                 {
                   for(uint k = 0; k < Q_size1; k++)
                     {

                      // h = data_Q(i+1, k);
                       NumericT h = data_Q[viennacl::column_major::mem_index((i + 1) * Q_inc1 + Q_start1, k * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)];

                       //Q(i+1, k) = tmp2[i] * Q(i, k) + tmp1[i]*h;
                       data_Q[viennacl::column_major::mem_index((i + 1) * Q_inc1 + Q_start1, k * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] = tmp2[start2 + inc2 *i] *
                           data_Q[viennacl::column_major::mem_index(i  * Q_inc1 + Q_start1, k * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] + tmp1[i] * h;

                       //Q(i,   k) = tmp1[i] * Q(i, k) - tmp2[i]*h;
                       data_Q[viennacl::column_major::mem_index(i  * Q_inc1 + Q_start1, k * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] = tmp1[start1 + inc1 * i] *
                           data_Q[viennacl::column_major::mem_index(i  * Q_inc1 + Q_start1, k * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] - tmp2[start2 + inc2 * i]*h;
                     }
                 }
             }

         }




         template <typename NumericT, typename F, typename S1>
         void copy_vec(matrix_base<NumericT, F>& A,
                       vector_base<S1> & V,
                       vcl_size_t row_start,
                       vcl_size_t col_start,
                       bool copy_col
         )
         {
             typedef NumericT        value_type;

             value_type * data_A  = detail::extract_raw_pointer<value_type>(A);
             value_type * data_V = detail::extract_raw_pointer<value_type>(V);

             vcl_size_t A_start1 = viennacl::traits::start1(A);
             vcl_size_t A_start2 = viennacl::traits::start2(A);
             vcl_size_t A_inc1   = viennacl::traits::stride1(A);
             vcl_size_t A_inc2   = viennacl::traits::stride2(A);
             vcl_size_t A_size1  = viennacl::traits::size1(A);
             vcl_size_t A_size2  = viennacl::traits::size2(A);
             vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(A);
             vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(A);

             vcl_size_t start1 = viennacl::traits::start(V);
             vcl_size_t inc1   = viennacl::traits::stride(V);
             vcl_size_t size1  = viennacl::traits::size(V);


           if(copy_col)
           {
               if (detail::is_row_major(typename F::orientation_category()))
               {
#ifdef VIENNACL_WITH_OPENMP
               #pragma omp parallel for
#endif
                   for(vcl_size_t i = row_start; i < A_size1; i++)
                   {
                      data_V[i - row_start] = data_A[viennacl::row_major::mem_index(i * A_inc1 + A_start1, col_start * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
                   }
               }
               else
               {
#ifdef VIENNACL_WITH_OPENMP
               #pragma omp parallel for
#endif
                   for(vcl_size_t i = row_start; i < A_size1; i++)
                   {
                      data_V[i - row_start] = data_A[viennacl::column_major::mem_index(i * A_inc1 + A_start1, col_start * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
                   }
               }
           }
           else
           {
               if (detail::is_row_major(typename F::orientation_category()))
               {
#ifdef VIENNACL_WITH_OPENMP
               #pragma omp parallel for
#endif
                   for(vcl_size_t i = col_start; i < A_size1; i++)
                   {
                      data_V[i - col_start] = data_A[viennacl::row_major::mem_index(row_start * A_inc1 + A_start1, i * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
                   }
               }
               else
               {
#ifdef VIENNACL_WITH_OPENMP
               #pragma omp parallel for
#endif
                   for(vcl_size_t i = col_start; i < A_size1; i++)
                   {
                      data_V[i - col_start] = data_A[viennacl::column_major::mem_index(row_start * A_inc1 + A_start1, i * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
                   }
               }
            }
         }


    } // namespace host_based
  } //namespace linalg
} //namespace viennacl


#endif
