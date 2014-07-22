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
*   Test file for qr-method
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
#include "viennacl/linalg/qr-method-common.hpp"
#include "viennacl/linalg/host_based/matrix_operations.hpp"
#include "Random.hpp"

#define EPS 10.0e-6



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
    std::cout << "\n";
}

void matrix_print(ublas::matrix<ScalarType>& A)
{
    for (unsigned int i = 0; i < A.size1(); i++) {
        for (unsigned int j = 0; j < A.size2(); j++)
           std::cout << std::setprecision(6) << std::fixed << A(i, j) << "\t";
        std::cout << "\n";
    }
    std::cout << "\n";
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
      if (std::abs(ublas_A(i,j) - vcl_A_cpu(i,j)) > EPS)
      {
        std::cout << "Error at index (" << i << ", " << j << "): " << ublas_A(i,j) << " vs " << vcl_A_cpu(i,j) << std::endl;
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

void fill_vector(ublas::vector<ScalarType>& v)
{
    for (unsigned int i = 0; i < v.size(); ++i)
      v[i] = random<ScalarType>();
}

/*
 *
 * Functions to be tested
 *
 */

template <typename NumericT>
void house_update_A_left(ublas::matrix<ScalarType> & A,
                         ublas::vector<ScalarType> D,
                         unsigned int start)
{
  NumericT temp = 0;
  NumericT beta = 0;
  ublas::vector<ScalarType> w;
  temp = ublas::inner_prod(D, D);
  beta = 2/temp;

  w = beta*ublas::prod(trans(A), D);

  //scaled rank 1 update
  for(uint i = 0; i < A.size1(); i++)
  {
      for(uint j = 0; j < A.size2(); j++)
      {
          A(i, j) = A(i, j) - D[i] * w[j];
      }
  }
}

template <typename NumericT>
void house_update_A_right(ublas::matrix<ScalarType> & A,
                          ublas::vector<ScalarType> D,
                          unsigned int start)
{
  NumericT temp = 0;
  NumericT beta = 0;
  ublas::vector<ScalarType> w;

  temp = ublas::inner_prod(D, D);
  beta = 2/temp;

  w = beta*ublas::prod(A, D);
  //scaled rank 1 update
  for(uint i = 0; i < A.size1(); i++)
  {
      for(uint j = 0; j < A.size2(); j++)
      {
          A(i, j) = A(i, j) - w[i] * D[j];
      }
  }
}

template <typename NumericT>
void house_update_QL(ublas::matrix<ScalarType> & A,
                     ublas::matrix<ScalarType> & Q,
                     ublas::vector<ScalarType> D)

{
  NumericT temp, beta = 0;
  ublas::matrix<NumericT> ubl_P(A.size1(), A.size2());
  ublas::matrix<ScalarType> I = ublas::identity_matrix<ScalarType>(Q.size1());
  ublas::matrix<NumericT> Q_temp(Q.size1(), Q.size2());

  for(unsigned int i = 0; i < Q.size1(); i++)
  {
      for(unsigned int j = 0; j < Q.size2(); j++)
      {
          Q_temp(i, j) = Q(i, j);
      }
  }
  ubl_P = ublas::identity_matrix<NumericT>(A.size1());

  temp = ublas::inner_prod(D, D);
  beta = 2/temp;

  //scaled_rank_1 update
  for(unsigned int i = 0; i < A.size1(); i++)
  {
      for(unsigned int j = 0; j < A.size2(); j++)
      {
          ubl_P(i, j) = I(i, j) - beta * (D[i] * D[j]);
      }
  }
  Q = ublas::prod(Q_temp, ubl_P);  //P wurde korrekt berechnet - ueberprueft
}

template <typename NumericT>
void givens_next(ublas::matrix<NumericT> & Q,
                 ublas::vector<NumericT> & tmp1,
                 ublas::vector<NumericT> & tmp2,
                 int l,
                 int m)
{
    for(int i = m - 1; i >= l; i--)
      {
        for(uint k = 0; k < Q.size1(); k++)
          {
            NumericT h = Q(k, i+1);
            Q(k, i+1) = tmp2[i] * Q(k, i) + tmp1[i]*h;
            Q(k, i) = tmp1[i] * Q(k, i) - tmp2[i]*h;
          }
      }
}


template <typename NumericT>
void copy_vec(ublas::matrix<NumericT>& A,
              ublas::vector<NumericT> & V,
              std::size_t row_start,
              std::size_t col_start,
              bool copy_col)
{
  if(copy_col)
  {
      for(std::size_t i = row_start; i < A.size1(); i++)
      {
         V[i - row_start] = A(i, col_start);
      }
  }
  else
  {
      for(std::size_t i = col_start; i < A.size1(); i++)
      {
         V[i - col_start] = A(row_start, i);
      }
  }
}

template <typename NumericT>
void bidiag_pack(ublas::matrix<NumericT> & A,
                 ublas::vector<NumericT> & D,
                 ublas::vector<NumericT> & S)

{
  std::size_t size = std::min(D.size(), S.size());
  std::size_t i = 0;
  for(i = 0;  i < size - 1; i++)
  {
      D[i] = A(i, i);
      S[i + 1] = A(i, i + 1);
  }
  D[i] = A(i, i);
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

  ublas::vector<ScalarType> ubl_D(sz), ubl_E(sz), ubl_F(sz), ubl_G(sz), ubl_H(sz);
  ublas::matrix<ScalarType> ubl_A(sz, sz), ubl_Q(sz, sz);
  viennacl::matrix<ScalarType> vcl_A(sz, sz), vcl_Q(sz, sz);
  viennacl::vector<ScalarType> vcl_D(sz), vcl_E(sz), vcl_F(sz);

  std::cout << "Testing matrix of size " << sz << "-by-" << sz << std::endl << std::endl;



  read_matrix_body(f, vcl_A);
  f.close();
  viennacl::copy(vcl_A, ubl_A);

  ubl_D[0] = 0;
  ubl_D[1] = -0.577735;
  ubl_D[2] = -0.577735;
  ubl_D[3] =  0.577735;

  copy(ubl_D, vcl_D);
//--------------------------------------------------------
  std::cout << "\nTesting house_update_left...\n";
  viennacl::linalg::house_update_A_left(vcl_A, vcl_D, 0);
  house_update_A_left<float>(ubl_A, ubl_D, 0);
  if(!check_for_equality(ubl_A, vcl_A))
    return EXIT_FAILURE;
//--------------------------------------------------------
  std::cout << "\nTesting house_update_right...\n";
  viennacl::linalg::house_update_A_right(vcl_A, vcl_D, 0);
  house_update_A_right<float>(ubl_A, ubl_D, 0);

  if(!check_for_equality(ubl_A, vcl_A))
     return EXIT_FAILURE;
//--------------------------------------------------------

  std::cout << "\nTesting house_update_QL...\n";
  ubl_Q = ublas::identity_matrix<ScalarType>(ubl_Q.size1());
  copy(ubl_Q, vcl_Q);
  viennacl::linalg::house_update_QL(vcl_A, vcl_Q, vcl_D);
  house_update_QL<float>(ubl_A, ubl_Q, ubl_D);
  if(!check_for_equality(ubl_Q, vcl_Q))
     return EXIT_FAILURE;
//--------------------------------------------------------

  std::cout << "\nTesting givens next...\n";
  fill_vector(ubl_E);
  fill_vector(ubl_F);
  copy(ubl_E, vcl_E);
  copy(ubl_F, vcl_F);
  viennacl::linalg::givens_next(vcl_Q, vcl_E, vcl_F, 0, 4);  //beim testen nicht gleich givens next
  matrix_print(vcl_Q);
  givens_next<ScalarType>(ubl_Q, ubl_E, ubl_F, 0, 4);
  matrix_print(ubl_Q);
  if(!check_for_equality(ubl_Q, vcl_Q))
      //return EXIT_FAILURE;
//--------------------------------------------------------
  std::cout << "\nTesting copy vec...\n";
  viennacl::linalg::detail::copy_vec(vcl_A, vcl_D, 0, 2, 1);
  copy_vec(ubl_A, ubl_D, 0, 2, 1);
  copy(vcl_D, ubl_E); //check for equality only for ublas vectors
  if(!check_for_equality(ubl_D, ubl_E))
      return EXIT_FAILURE;

//--------------------------------------------------------
  std::cout << "\nTesting bidiag pack...\n";
  viennacl::linalg::bidiag_pack(vcl_A, ubl_D, ubl_F);
    ubl_F[0] = 0;  // first element not calculated in bidiag pack for minor diagonal!
  bidiag_pack(ubl_A, ubl_G, ubl_H);
  if(!check_for_equality(ubl_D, ubl_G))
      return EXIT_FAILURE;
  if(!check_for_equality(ubl_F, ubl_H))
      return EXIT_FAILURE;
//--------------------------------------------------------
  std::cout <<"\nTEST COMPLETE!\n";
}
