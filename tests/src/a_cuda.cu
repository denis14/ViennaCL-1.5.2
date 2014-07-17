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
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/cuda/matrix_operations_col.hpp"


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
    std::cout << "matrix_print!\n";
    for (unsigned int i = 0; i < A.size1(); i++) {
        for (unsigned int j = 0; j < A.size2(); j++)
           std::cout << A(i, j) << "\t";
        std::cout << "\n";
    }
}

void vector_print(ublas::vector<ScalarType>& v )
{

    std::cout << "vector_print!\n";
    for (unsigned int i = 0; i < v.size(); i++)
      std::cout << v(i) << "\t";
    std::cout << "\n";
}
int main()
{
  //ublas::vector<ScalarType> D = ublas::scalar_vector<ScalarType>(sz, 0);
  //ublas::vector<ScalarType> E = ublas::scalar_vector<ScalarType>(sz, 0);
  
  //std::size_t sz = 4;
  // read file
  //std::fstream f("../../examples/testdata/eigen/symm1.example", std::fstream::in);
  //read size of input matrix
  //read_matrix_size(f, sz);
  
  //viennacl::matrix<ScalarType> A_input(4, 4);

  //read_matrix_body(f, A_input);
  //f.close();
  //matrix_print(A_input);
  
  viennacl::linalg::cuda::bidiag_pack_kernel<<<128, 128>>>(10);
  std::cout << "Testdata wurde gelesen! git" << std::endl;
  //vector_print(D);
  //matrix_print(A_input);

}


