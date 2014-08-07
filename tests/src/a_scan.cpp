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

//#define VIENNACL_WITH_UBLAS

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

//#include <boost/numeric/ublas/matrix.hpp>
//#include <boost/numeric/ublas/matrix_proxy.hpp>
//#include <boost/numeric/ublas/matrix_expression.hpp>
//#include <boost/numeric/ublas/matrix_sparse.hpp>
//#include <boost/numeric/ublas/vector.hpp>
//#include <boost/numeric/ublas/operation.hpp>
//#include <boost/numeric/ublas/vector_expression.hpp>
#include "viennacl/linalg/qr-method-common.hpp"
#include "viennacl/linalg/host_based/matrix_operations.hpp"
#include "Random.hpp"

#define EPS 10.0e-3



namespace ublas = boost::numeric::ublas;
typedef float     ScalarType;



void matrix_print(viennacl::matrix<ScalarType>& A_orig)
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



void vector_print(viennacl::vector<ScalarType>& v )
{
  ublas::vector<ScalarType> v_ublas = ublas::vector<ScalarType>(v.size(), 0);
  copy(v, v_ublas);

  for (unsigned int i = 0; i < v.size(); i++)
      std::cout << std::setprecision(6) << std::fixed << v_ublas(i) << "\t";
    std::cout << "\n";
}


void fill_vector(viennacl::vector<ScalarType>& v)
{
   // for (unsigned int i = 0; i < v.size(); ++i)
      //v[i] =  random<ScalarType>();
      //v[i] =  i;
      v[0]  = 2;
      v[1]  = 1;
      v[2]  = 3;
      v[3]  = 1;
      v[4]  = 0;
      v[5]  = 4;
      v[6]  = 1;
      v[7]  = 2;
      v[8]  = 0;
      v[9]  = 3;
      v[10] = 1;
      v[11] = 2;
      v[12] = 5;
      v[13] = 3;
      v[14] = 1;
      v[15] = 2;
}


/*
 *
 * ------------Functions to be tested---------------
 *
 */



void test_scan()
{
  std::cout << "test started..." << std::endl;
  unsigned int sz = 16;




  viennacl::vector<ScalarType> vec1(sz), vec2(sz);

  fill_vector(vec1);
  viennacl::linalg::cuda::inclusive_scan(vec1, vec2);

  vector_print(vec1);
  vector_print(vec2);

//--------------------------------------------------------

//--------------------------------------------------------
}

int main()
{

  std::cout << std::endl << "Test scan" << std::endl;
  test_scan();

  std::cout << std::endl <<"--------TEST SUCCESSFULLY COMPLETED----------" << std::endl;
}