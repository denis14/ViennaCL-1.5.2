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
Solutions for testdata were generated with Scilab line:

M=fscanfMat('nsm1.example');e=spec(M);e=gsort(e);rr=real(e);ii=imag(e);e=cat(1, rr, ii); s=strcat(string(e), ' ');write('tmp', s);
*/

#ifndef NDEBUG
  #define NDEBUG
#endif

//#define VIENNACL_DEBUG_ALL
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/qr-method.hpp"

#include <examples/benchmarks/benchmark-utils.hpp>

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

template <typename MatrixLayout>
void test_eigen(const std::string& fn)
{
    /*
     *                      Tutorial for the qr-method
     *
     * The eigenvalues and eigenvectors of a symmetric 9 by 9 matrix are calculated
     * by the QR-method.
     *
     */
    std::cout << "Reading..." << "\n";
    std::size_t sz;
    // read file
    std::fstream f(fn.c_str(), std::fstream::in);
    //read size of input matrix
    read_matrix_size(f, sz);
    std::cout << "Testing matrix of size " << sz << "-by-" << sz << std::endl;

    viennacl::matrix<ScalarType, MatrixLayout> A_input(sz, sz), A_ref(sz, sz), Q(sz, sz);
    ublas::vector<ScalarType> eigen_re = ublas::scalar_vector<ScalarType>(sz, 0);


    read_matrix_body(f, A_input);

    std::cout <<"\nInput matrix: \n";
    matrix_print(A_input);

    read_vector_body(f, eigen_re);

    std::cout <<"\n Input vector: \n";
    vector_print(eigen_re);

    f.close();

    A_ref = A_input;

    std::cout << "\nStarting QR-method \n";
    std::cout << "Calculation..." << "\n";
    viennacl::linalg::qr_method_sym(A_input, Q, eigen_re);
    std::cout <<"\nEigenvalues - in the main diagonal\n";
    matrix_print(A_input);
    std::cout <<"\nEigenvectors - each column is one eigenvector\n";
    matrix_print(Q);

}

int main()
{

  test_eigen<viennacl::row_major>("../examples/testdata/eigen/symm5.example");

  std::cout << std::endl;
  std::cout << "------- Tutorial completed --------" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
