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
  bool randomValues = true;


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
       diagonal[i] = ((NumericT)(i % 37)) - 4.5f;
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
    bool test_result = true;
    unsigned int time_index = 0;
    std::vector<double> av_times(500);
    std::vector<unsigned int> mat_sizes(500);

    for( unsigned int mat_size = 16;
         mat_size < 30000;
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
      std::cout << mat_sizes[i] << "\t" << av_times[i] << std::endl;

    values_save(av_times, mat_sizes, time_index);



/*



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
    std::vector<NumericT> eigenvalues_bisect_cpu(mat_size);

    // for tql2 algorithm
    //viennacl::matrix<NumericT, viennacl::row_major> Q = viennacl::identity_matrix<NumericT>(mat_size);


    // -------Start the bisection algorithm------------
    std::cout << "Start the bisection algorithm" << std::endl;
    std::cout << "Matrix size: " << mat_size << std::endl;

    unsigned int iterations = 20;
    unsigned int max_eigen = 0, max_eigen_abs = 0;
    double time_all = 0.0;
    for(unsigned int i = 0; i < iterations; i++)
    {
      initInputData(diagonal, superdiagonal, mat_size);

      Timer timer;
      timer.start();

      // bisection - gpu
      bResult = viennacl::linalg::bisect(diagonal, superdiagonal, eigenvalues_bisect);

      //---Run the tql2 algorithm-----------------------------------
      //viennacl::linalg::tql1<NumericT>(mat_size, diagonal, superdiagonal);
      //bResult = true;

      // Run the bisect algorithm for CPU only
      //eigenvalues_bisect_cpu = viennacl::linalg::bisect(diagonal, superdiagonal);
      // bResult = true;



    //  MatrixXd A = MatrixXd::Random(mat_size, mat_size);
      //std::cout << "Here is a random " << mat_size << " matrix, A:" << std::endl << A << std::endl << std::endl;
  //    EigenSolver<MatrixXd> es(A);
  //    es.eigenvalues();
//      es.eigenvectors();
      //std::cout << "The eigenvalues of A are:" << std::endl << es.eigenvalues() << std::endl;

      // Exit if an error occured during the execution of the algorithm
      if (bResult == false)
       return false;
      time_all += timer.get() * 1000;

    }
    /*
    for(unsigned int n = 1; n < eigenvalues_bisect.size(); n++)
    {
      for(unsigned int m = 0; m < n; m++)
      {
        if(eigenvalues_bisect[m] == eigenvalues_bisect[n])
          max_eigen++;
      }
      max_eigen_abs = max(max_eigen_abs, max_eigen);
      max_eigen = 0;
    }
    */
    double time_average = time_all / (double)iterations;

    std::cout << "Time all: \t" << time_all << "ms" << "\taverage Time:\t" << time_average << "ms" <<  " max_eigen:\t" << max_eigen_abs << std::endl;
    av_times[time_index]  = time_average;


  return bResult;

}
