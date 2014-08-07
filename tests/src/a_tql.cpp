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

#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <iomanip>

#include "viennacl/linalg/qr-method.hpp"
#include "viennacl/linalg/qr-method-common.hpp"
#include "viennacl/linalg/host_based/matrix_operations.hpp"
#include "Random.hpp"

#define EPS 10.0e-5



namespace ublas = boost::numeric::ublas;
typedef float     ScalarType;


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


/**
 * Test the tql2 algorithm for symmetric tridiagonal matrices.
 */

template <typename MatrixLayout>
void test_qr_method_sym()
{
  std::size_t sz = 10;

  viennacl::matrix<ScalarType, MatrixLayout> Q = viennacl::identity_matrix<ScalarType>(sz);
  ublas::vector<ScalarType> d(sz), e(sz), d_ref(sz), e_ref(sz);

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

  if(!test_eigen_val_vec<MatrixLayout>(Q, d, d_ref, e_ref))
     exit(EXIT_FAILURE);
}

int main()
{

  std::cout << std::endl << "Testing QL algorithm for symmetric tridiagonal row-major matrices..." << std::endl;
  test_qr_method_sym<viennacl::row_major>();

  std::cout << std::endl << "Testing QL algorithm for symmetric tridiagonal column-major matrices..." << std::endl;
  test_qr_method_sym<viennacl::column_major>();

  std::cout << std::endl <<"--------TEST SUCCESSFULLY COMPLETED----------" << std::endl;
}
