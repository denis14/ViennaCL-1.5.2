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

/*
*
*   Test file for tql-algorithm
*
*/

// include necessary system headers
#include <iostream>

#ifndef NDEBUG
  #define NDEBUG
#endif

#define VIENNACL_WITH_UBLAS

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"


//#include "viennacl/linalg/lanczos.hpp"
#include "viennacl/io/matrix_market.hpp"
// Some helper functions for this tutorial:
#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <iomanip>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include "viennacl/linalg/qr-method.hpp"
#include "viennacl/linalg/qr-method-common.hpp"
#include "viennacl/linalg/host_based/matrix_operations.hpp"
#include "Random.hpp"

#define EPS 10.0e-5



namespace ublas = boost::numeric::ublas;
typedef float     ScalarType;


template <typename MatrixLayout>
void matrix_print(viennacl::matrix<ScalarType, MatrixLayout>& A_orig)
{
    ublas::matrix<ScalarType> A(A_orig.size1(), A_orig.size2());
    viennacl::copy(A_orig, A);
    for (unsigned int i = 0; i < A.size1(); i++) {
        for (unsigned int j = 0; j < A.size2(); j++)
           std::cout << std::setprecision(6) << std::fixed << A(i, j) << "\t";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void vector_print(ublas::vector<ScalarType>& v )
{
    for (unsigned int i = 0; i < v.size(); i++)
      std::cout << std::setprecision(6) << std::fixed << v(i) << "\t";
    std::cout << "\n";
}


template <typename MatrixType, typename VCLMatrixType>
bool check_for_equality(MatrixType const & ublas_A, VCLMatrixType const & vcl_A)
{
  typedef typename MatrixType::value_type   value_type;

  ublas::matrix<value_type> vcl_A_cpu(vcl_A.size1(), vcl_A.size2());
  viennacl::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
  viennacl::copy(vcl_A, vcl_A_cpu);

  for (std::size_t i=0; i<ublas_A.size1(); ++i)
  {
    for (std::size_t j=0; j<ublas_A.size2(); ++j)
    {
      if (std::abs(ublas_A(i,j) - vcl_A_cpu(i,j)) > EPS * std::max(std::abs(ublas_A(i, j)), std::abs(vcl_A_cpu(i, j))))
      {
        std::cout << "Error at index (" << i << ", " << j << "): " << ublas_A(i,j) << " vs. " << vcl_A_cpu(i,j) << std::endl;
        std::cout << std::endl << "TEST failed!" << std::endl;
        return false;
      }
    }
  }
  std::cout << "PASSED!" << std::endl;
  return true;
}

template <typename VectorType>
bool check_for_equality(VectorType const & ublas_A, VectorType const & ublas_B)
{

  for (std::size_t i=0; i<ublas_A.size(); ++i)
  {
      if (std::abs(ublas_A(i) - ublas_B(i)) > EPS)
      {
        std::cout << "Error at index (" << i << "): " << ublas_A(i) << " vs " <<ublas_B(i) << std::endl;
        std::cout << std::endl << "TEST failed!" << std::endl;
        return false;
      }
  }
  std::cout << "PASSED!" << std::endl;
  return true;
}

// Test the eigenvectors
// Perform the multiplication (T - lambda * I) * Q, with the original tridiagonal matrx T, the
// eigenvalues lambda and the eigenvectors in Q. The result has to be 0.

template <typename MatrixLayout>
bool test_eigen_val_vec(viennacl::matrix<ScalarType, MatrixLayout> & Q,
                      ublas::vector<ScalarType> & eigenvalues,
                      ublas::vector<ScalarType> & d,
                      ublas::vector<ScalarType> & e)
{
  unsigned int Q_size = Q.size2();
  ScalarType value = 0;

  for(unsigned int j = 0; j < Q_size; j++)
  {
    // calculate first row
    value = (d[0]- eigenvalues[j]) * Q(0, j) + e[1] * Q(1, j);
    if(value > EPS)
      return false;

    // calcuate inner rows
    for(unsigned int i = 1; i < Q_size - 1; i++)
    {
        value = e[i] * Q(i - 1, j) + (d[i]- eigenvalues[j]) * Q(i, j) + e[i + 1] * Q(i + 1, j);
        if(value > EPS)
          return false;
    }

    // calculate last row
    value = e[Q_size - 1] * Q(Q_size - 2, j) + (d[Q_size - 1] - eigenvalues[j]) * Q(Q_size - 1, j);
     if(value > EPS)
       return false;
  }
  return true;
}




template <typename MatrixLayout>
void test_qr_method_sym()
{
  std::size_t sz = 10;

  viennacl::matrix<ScalarType, MatrixLayout> Q = viennacl::identity_matrix<ScalarType>(sz), A_ref(sz, sz);
  ublas::matrix<ScalarType> A_ref_ublas(sz, sz), A_input_ublas(sz, sz), Q_ublas(sz, sz), result1(sz, sz), result2(sz, sz);

  ublas::vector<ScalarType> d(sz), e(sz), d_ref(sz), e_ref(sz);
  ublas::matrix<ScalarType> ubl_A(sz, sz), ubl_Q(sz, sz);




  std::cout << "Testing matrix of size " << sz << "-by-" << sz << std::endl << std::endl;

  // Initialize diagonal and superdiagonal elements
  d[0] = 1; e[0] = 0;
  d[1] = 2; e[1] = 4;
  d[2] =-4; e[2] = 5;
  d[3] = 6; e[3] = 1;
  d[4] = 3; e[4] = 2;
  d[5] = 4; e[5] =-3;
  d[6] = 7; e[6] = 5;
  d[7] = 9; e[7] = 1;
  d[8] = 3; e[8] = 5;
  d[9] = 8; e[9] = 2;

  d_ref = d;
  e_ref = e;

//--------------------------------------------------------
  viennacl::linalg::tql2(Q, d, e);

  std::cout << "Eigenvalues: " << std::endl;
  vector_print(d);
  std::cout << std::endl << "Eigenvectors: " << std::endl << std::endl;
  matrix_print(Q);
  std::cout << "Eigenvectors correct? " << test_eigen_val_vec<MatrixLayout>(Q, d, d_ref, e_ref) << std::endl;
}

int main()
{

  std::cout << std::endl << "Testing QL algorithm for symmetric tridiagonal row-major matrices..." << std::endl;
  test_qr_method_sym<viennacl::row_major>();

  std::cout << std::endl << "Testing QL algorithm for symmetric tridiagonal column-major matrices..." << std::endl;
  test_qr_method_sym<viennacl::column_major>();

  std::cout << std::endl <<"--------TEST SUCCESSFULLY COMPLETED----------" << std::endl;
}
