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

#ifndef NDEBUG
  #define NDEBUG
#endif

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>

//#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/qr-method.hpp"

//#include <examples/benchmarks/benchmark-utils.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

namespace ublas = boost::numeric::ublas;

typedef float ScalarType;

const ScalarType EPS = 0.0001f;

void read_matrix_size(std::fstream& f, std::size_t& sz)
{
    if(!f.is_open())
    {
        throw std::invalid_argument("File is not opened");
    }

    f >> sz;
}

template <typename MatrixLayout>
void read_matrix_body(std::fstream& f, viennacl::matrix<ScalarType, MatrixLayout>& A)
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

void read_vector_body(std::fstream& f, ublas::vector<ScalarType>& v) {
    if(!f.is_open())
        throw std::invalid_argument("File is not opened");

    for(std::size_t i = 0; i < v.size(); i++)
    {
            ScalarType val = 0.0;
            f >> val;
            v[i] = val;
    }
}


template <typename MatrixLayout>
void matrix_print(viennacl::matrix<ScalarType, MatrixLayout>& A_orig)
{
    ublas::matrix<ScalarType> A(A_orig.size1(), A_orig.size2());
    viennacl::copy(A_orig, A);
    for (unsigned int i = 0; i < A.size1(); i++) {
        for (unsigned int j = 0; j < A.size2(); j++)
           std::cout << A(i, j) << "\t";
        std::cout << "\n";
    }
}

void vector_print(ublas::vector<ScalarType>& v_ublas )
{
  //ublas::vector<ScalarType> v_ublas = ublas::vector<ScalarType>(v.size(), 0);
  //copy(v, v_ublas);

  for (unsigned int i = 0; i < v_ublas.size(); i++)
      std::cout << std::setprecision(6) << std::fixed << v_ublas(i) << "\t";
    std::cout << "\n";
}
