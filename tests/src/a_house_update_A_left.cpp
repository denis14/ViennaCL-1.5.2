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
*   Tutorial: Calculation of eigenvalues using Lanczos' method (lanczos.cpp and lanczos.cu are identical, the latter being required for compilation using CUDA nvcc)
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


#include "viennacl/linalg/lanczos.hpp"
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

namespace ublas = boost::numeric::ublas;
typedef float     ScalarType;


void read_matrix_size(std::fstream& f, std::size_t& sz)
{
    if(!f.is_open())
    {
        throw std::invalid_argument("File is not opened");
    }

    f >> sz;
}

void read_matrix_body(std::fstream& f, viennacl::matrix<ScalarType>& A)
{
    if(!f.is_open())
    {
        throw std::invalid_argument("File is not opened");
    }

    boost::numeric::ublas::matrix<ScalarType> h_A(A.size1(), A.size2());

    for(std::size_t i = 0; i < h_A.size1(); i++) {
        for(std::size_t j = 0; j < h_A.size2(); j++) {
            ScalarType val = 0.0;
            f >> val;
            h_A(i, j) = val;
        }
    }

    viennacl::copy(h_A, A);
}

void matrix_print(viennacl::matrix<ScalarType>& A_orig)
{
    ublas::matrix<ScalarType> A(A_orig.size1(), A_orig.size2());
    viennacl::copy(A_orig, A);
    for (unsigned int i = 0; i < A.size1(); i++) {
        for (unsigned int j = 0; j < A.size2(); j++)
           std::cout << std::setprecision(6) << std::fixed << A(i, j) << "\t";
        std::cout << "\n";
    }
}

void vector_print(ublas::vector<ScalarType>& v )
{
    for (unsigned int i = 0; i < v.size(); i++)
      std::cout << std::setprecision(5) << std::fixed << v(i) << "\t";
    std::cout << "\n";
}

void vector_print(viennacl::vector<ScalarType>& v )
{
  ublas::vector<ScalarType> v_ublas = ublas::vector<ScalarType>(v.size(), 0);
  copy(v, v_ublas);

  for (unsigned int i = 0; i < v.size(); i++)
      std::cout << std::fixed << v_ublas(i) << "\t";
    std::cout << "\n";
}


int main()
{
/*
 * Programm zum Testen der einzelnen OpenMP Funktionen
 * vom Ordner /build aus aufrufen!
 *
 * Matrix wird automatisch eingelesen, dann wird die entsprechende
 * Funktion  (z.B. house_update_A_right ) darauf angewendet, dann
 * wird sie wieder ausgegeben
 *
 */
  std::cout << "Reading..." << "\n";
  std::size_t sz;
  // read file
  std::fstream f("../../examples/testdata/eigen/symm1.example", std::fstream::in);
  //read size of input matrix
  read_matrix_size(f, sz);
  ublas::vector<ScalarType> D = ublas::scalar_vector<ScalarType>(sz, 0);
  ublas::vector<ScalarType> E = ublas::scalar_vector<ScalarType>(sz, 0);
  viennacl::matrix<ScalarType> A(sz, sz), A_ref(sz, sz), Q(sz, sz);
  std::cout << "Testing matrix of size " << sz << "-by-" << sz << std::endl << std::endl;

  read_matrix_body(f, A);
  f.close();

  std::cout << "Original Matrix A:\n";
  matrix_print(A);


  D[0] = 0;
  D[1] = -0.577735;
  D[2] = -0.577735;
  D[3] =  0.577735;

  viennacl::vector<ScalarType> vcl_D(sz);
  copy(D, vcl_D);
  std::cout << "\nExecuting house_update_left...\n";
  viennacl::linalg::house_update_A_left(A, vcl_D, 0);
  //viennacl::linalg::house_update_A_right(A, vcl_D, 0);

  std::cout << "\nVector D: \n";
  vector_print(vcl_D);
  std::cout << "\nMatrix A: \n";
  matrix_print(A);
  //std::cout << "\nVector E: \n";
  //vector_print(E);
}


