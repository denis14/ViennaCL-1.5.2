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
#include "viennacl/linalg/tql2.hpp"

#include <examples/benchmarks/benchmark-utils.hpp>
//#include <Eigen/Eigenvalues>


//using namespace Eigen;

#define EPS 10.0e-4

typedef float NumericT;
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(const int mat_size, std::vector<double> &av_times, unsigned int time_index);



////////////////////////////////////////////////////////////////////////////////
/// \brief initInputData   Initialize the diagonal and superdiagonal elements of
///                        the matrix
/// \param diagonal        diagonal elements of the matrix
/// \param superdiagonal   superdiagonal elements of the matrix
/// \param mat_size        Dimension of the matrix
///
void
initInputData(std::vector<NumericT> &diagonal, std::vector<NumericT> &superdiagonal, const unsigned int mat_size)
{
 
  srand(time(NULL));
  bool randomValues = false;
  
  
  if (randomValues == true)
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
       diagonal[i] = ((NumericT)(i % 8)) - 4.5f;
       superdiagonal[i] = ((NumericT)(i % 5)) - 4.5f;
    }
  }
  // the first element of s is used as padding on the device (thus the
  // whole vector is copied to the device but the kernels are launched
  // with (s+1) as start address
  superdiagonal[0] = 0.0f; 
}

bool values_save(std::vector<double> &av_times, std::vector<unsigned int> &mat_sizes, unsigned int num_tests)
{

    FILE *datei;
    long i;

    if(!(datei=fopen("../../execution_times_matrices.dat","w")))
    {
      fprintf(stderr,"Error: file access!\n");
      return false;
    }

    for(i=0; i < num_tests; i++)
    {
      fprintf(datei,"%i\t %7.3f\n", mat_sizes[i], av_times[i]);
    }

    fclose(datei);
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    bool test_result = false;
    unsigned int time_index = 0;
    std::vector<double> av_times(500);
    std::vector<unsigned int> mat_sizes(500);

    for( unsigned int mat_size = 11;
         mat_size < 10000;
         mat_size = mat_size * 1.15, time_index++)
      {
      test_result = runTest(mat_size, av_times, time_index);
      std::cout << "Matrix_size = \t" << mat_size << std::endl;
      mat_sizes[time_index] = mat_size;


      if(test_result == true)
      {
        std::cout << "Test Succeeded!" << std::endl << std::endl;
      }
      else
      {
        std::cout << "---TEST FAILED---" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    for(unsigned int i = 0; i < time_index; i++)
      std::cout << "i: " << i << "\tMat_size " << mat_sizes[i] << "\ttime:\t" << av_times[i] << " ms" << std::endl;

    values_save(av_times, mat_sizes, time_index);

/*
    MatrixXd A = MatrixXd::Random(6,6);
    std::cout << "Here is a random 6x6 matrix, A:" << std::endl << A << std::endl << std::endl;
    EigenSolver<MatrixXd> es(A);
    std::cout << "The eigenvalues of A are:" << std::endl << es.eigenvalues() << std::endl;
    std::cout << "The matrix of eigenvectors, V, is:" << std::endl << es.eigenvectors() << std::endl << std::endl;



    // run test for small matrix
    test_result = runTest(230);
    if(test_result == true)
    {
      std::cout << std::endl << "---TEST SUCCESSFULLY COMPLETED---" << std::endl;
      exit(EXIT_SUCCESS);
    }
    else
    {
      std::cout << "---TEST FAILED---" << std::endl;
      exit(EXIT_FAILURE);
    }
*/

}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test
////////////////////////////////////////////////////////////////////////////////
bool
runTest(const int mat_size, std::vector<double> &av_times, unsigned int time_index)
{
    bool bResult = false;
    std::vector<NumericT> diagonal(mat_size);
    std::vector<NumericT> superdiagonal(mat_size);
    std::vector<NumericT> eigenvalues_bisect(mat_size);

    // -------------Initialize data-------------------
    // Fill the diagonal and superdiagonal elements of the vector


    // -------Start the bisection algorithm------------
    std::cout << "Start the bisection algorithm" << std::endl;
    std::cout << "Matrix size: " << mat_size << std::endl;

    unsigned int iterations = 10;
    double time_all = 0.0;
    for(unsigned int i = 0; i < iterations; i++)
    {
      initInputData(diagonal, superdiagonal, mat_size);

      Timer timer;
      timer.start();
      bResult = viennacl::linalg::bisect(diagonal, superdiagonal, eigenvalues_bisect);
      // Exit if an error occured during the execution of the algorithm
      if (bResult == false)
       return false;
      time_all += timer.get() * 1000;
    }
    double time_average = time_all / iterations;

    std::cout << "Time all: \t" << time_all << "ms" << "\taverage Time:\t" << time_average << "ms" << std::endl;
    av_times[time_index]  = time_average;


  return bResult;
    
}
