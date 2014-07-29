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

//#define VIENNACL_DEBUG_ALL
#include "vec_mat_io.hpp"


#include "viennacl/linalg/qr-method.hpp"

namespace ublas = boost::numeric::ublas;

typedef float ScalarType;

template <typename MatrixLayout>
void initialize(viennacl::matrix<ScalarType, MatrixLayout>& A, ublas::vector<ScalarType>& v);



template <typename MatrixLayout>
void qr_method()
{
    /*
     *                      Tutorial for the qr-method
     *
     * The eigenvalues and eigenvectors of a symmetric 9 by 9 matrix are calculated
     * by the QR-method.
     *
     */

    std::cout << "Testing matrix of size " << 9 << "-by-" << 9 << std::endl;

    viennacl::matrix<ScalarType, MatrixLayout> A_input(9,9);
    viennacl::matrix<ScalarType, MatrixLayout> Q(9, 9);
    ublas::vector<ScalarType> eigenvalues_ref(9);
    ublas::vector<ScalarType> eigenvalues = ublas::scalar_vector<ScalarType>(9, 0);

    initialize(A_input, eigenvalues_ref);  //initialize with data for tutorial

    std::cout <<"Input matrix: \n";
    matrix_print(A_input);

    std::cout <<"\nEigenvalues of input matrix: \n";
    vector_print(eigenvalues_ref);

    std::cout << "\nStarting QR-method \n";
    std::cout << "Calculation..." << "\n";

    /*
     * Call function qr_method_sym to calculate eigenvalues and eigenvectors
     * Parameters:
     *       A_input     - input matrix to find eigenvalues and eigenvectors from
     *       Q           - matrix, where the calculated eigenvectors will be stored in
     *       eigenvalues - vector, where the calculated eigenvalues will be stored in
     */

    viennacl::linalg::qr_method_sym(A_input, Q, eigenvalues);

    std::cout <<"\nEigenvalues:\n";
    vector_print(eigenvalues);
    std::cout <<"\nReference eigenvalues: \n";
    vector_print(eigenvalues_ref);
    std::cout <<"\nEigenvectors - each column is an eigenvector\n";

    matrix_print(Q);

}

int main()
{

  qr_method<viennacl::row_major>();

  std::cout << std::endl;
  std::cout << "------- Tutorial completed --------" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

//initialize vector and matrix
template <typename MatrixLayout>
void initialize(viennacl::matrix<ScalarType, MatrixLayout>& A, ublas::vector<ScalarType>& v)
{
    ScalarType M[9][9] = {{4, 1, -2, 2, -7, 3, 9, -6, -2}, {1, -2, 0, 1, -1, 5, 4, 7, 3}, {-2, 0, 3, 2, 0, 3, 6, 1, -1},   {2, 1, 2, 1, 4, 5, 6, 7, 8},
               {-7, -1, 0, 4, 5, 4, 9, 1, -8},  {3, 5, 3, 5, 4, 9, -3, 3, 3}, {9, 4, 6, 6, 9, -3, 3, 6, -7},   {-6, 7, 1, 7, 1, 3, 6, 2, 6},
               {-2, 3, -1, 8, -8, 3, -7, 6, 1}};

    for(int i = 0; i < 9; i++)
        for(int j = 0; j < 9; j++)
            A(i, j) = M[i][j];

    ScalarType V[9] = {12.6005, 19.5905, 8.06067, 2.95074, 0.223506, 24.3642, -9.62084, -13.8374, -18.3319};

    for(int i = 0; i < 9; i++)
        v[i] = V[i];
}
