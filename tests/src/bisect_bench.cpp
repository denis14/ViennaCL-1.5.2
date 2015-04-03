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

/* Computation of eigenvalues of a symmetric, tridiagonal matrix using
 * bisection.
 */

#ifndef NDEBUG
  #define NDEBUG
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>


// includes, project

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

#include "viennacl/linalg/bisect_gpu.hpp"
#include "viennacl/linalg/bisect.hpp"
#include "viennacl/linalg/tql2.hpp"

#include <examples/benchmarks/benchmark-utils.hpp>
//#include <Eigen/Eigenvalues>
//using namespace Eigen;

#define EPS 10.0e-4

typedef float NumericT;

////////////////////////////////////////////////////////////////////////////////
/// \brief initInputData   Initialize the diagonal and superdiagonal elements of
///                        the matrix
/// \param diagonal        diagonal elements of the matrix
/// \param superdiagonal   superdiagonal elements of the matrix
/// \param mat_size        Dimension of the matrix
///
void
initInputData(viennacl::vector<NumericT> &diagonal, viennacl::vector<NumericT> &superdiagonal, const unsigned int mat_size)
{

  srand(time(NULL));

#define RANDOM_VALUES 0
  if (RANDOM_VALUES == true)
  {
    // Initialize diagonal and superdiagonal elements with random values
    for (unsigned int i = 0; i < mat_size; ++i)
    {
        diagonal[i] =      static_cast<NumericT>(2.0 * (((double)rand()
                                     / (double) RAND_MAX) - 0.5));
        superdiagonal[i] = static_cast<NumericT>(2.0 * (((double)rand()
                                     / (double) RAND_MAX) - 0.5));
    }
  }
  else
  {
    // Initialize diagonal and superdiagonal elements with modulo values
    // This will cause in many multiple eigenvalues.
    for (unsigned int i = 0; i < mat_size; ++i)
    {
       diagonal[i] = ((NumericT)(i % 37)) - 4.5f;
       superdiagonal[i] = ((NumericT)(i % 5)) - 4.5f;
    }
  }
  // the first element of s is used as padding on the device (thus the
  // whole vector is copied to the device but the kernels are launched
  // with (s+1) as start address
  superdiagonal[0] = 0.0f;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test
////////////////////////////////////////////////////////////////////////////////
bool
runTest(const int mat_size, std::vector<double> &av_time_all, unsigned int time_index)
    {
    bool bResult = false;
    viennacl::vector<NumericT> diagonal(mat_size);
    viennacl::vector<NumericT> superdiagonal(mat_size);
    viennacl::vector<NumericT> eigenvalues_bisect(mat_size);
    std::vector<NumericT> eigenvalues_bisect_cpu(mat_size);


    // -------Start the bisection algorithm------------
    std::cout << "Matrix size: " << mat_size << std::endl;

    unsigned int iterations = 10;
    double time_all     = 0.0;

    for(unsigned int i = 0; i < iterations; i++)
    {
      initInputData(diagonal, superdiagonal, mat_size);

      Timer timer;
      timer.start();

      // bisection - gpu
      bResult = viennacl::linalg::bisect(diagonal, superdiagonal, eigenvalues_bisect);
      viennacl::backend::finish();     // sync
      //---Run the tql algorithm-----------------------------------
     // viennacl::linalg::tql1<NumericT>(mat_size, diagonal, superdiagonal);
     // bResult = true;

      // Run the bisect algorithm for CPU only
      //eigenvalues_bisect_cpu = viennacl::linalg::bisect(diagonal, superdiagonal);
      // bResult = true;

      time_all     += timer.get() * 1000;
      if (bResult == false)
       return false;

    }


    std::cout << "Time: \t" << time_all / (double)iterations << "ms" << std::endl << std::endl;

    av_time_all[time_index] = time_all / (double)iterations;

  return bResult;

}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    bool test_result = true;
    unsigned int time_index = 0;
    std::vector<double> av_time_all(500);
    std::vector<unsigned int> mat_sizes(500);

    for( unsigned int mat_size = 16;
         mat_size < 600;
         mat_size = mat_size * 1.15, time_index++)
      {
      test_result = runTest(mat_size, av_time_all, time_index);
      mat_sizes[time_index] = mat_size;


      if(test_result == true)
      {
        //std::cout << "Success!" << std::endl << std::endl;
      }
      else
      {
        std::cout << "---FAIL---" << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    std::cout << "Times" << std::endl;
    for(unsigned int i = 0; i < time_index; i++)
    {
      std::cout <<  mat_sizes[i] << "\t" << av_time_all[i] << std::endl;
    }




}

