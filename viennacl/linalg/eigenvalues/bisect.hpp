#ifndef VIENNACL_LINAL_BISECT_GPU
#define VIENNACL_LINAL_BISECT_GPU

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// includes, project


#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"


#include "viennacl/linalg/eigenvalues/config.hpp"
#include "viennacl/linalg/eigenvalues/structs.hpp"
#include "viennacl/linalg/eigenvalues/matlab.hpp"
#include "viennacl/linalg/eigenvalues/util.hpp"
#include "viennacl/linalg/eigenvalues/gerschgorin.hpp"
#include "viennacl/linalg/eigenvalues/bisect_large.hpp"
#include "viennacl/linalg/eigenvalues/bisect_small.cu"

namespace viennacl
{
  namespace linalg
  {
    bool bisect(const std::vector<float> & diagonal, const std::vector<float> & superdiagonal, std::vector<float> & eigenvalues, const unsigned int mat_size)
    {
        bool bCompareResult = false;
        // flag if the matrix size is due to explicit user request
        // desired precision of eigenvalues
        float  precision = 0.00001;
        char  *result_file = "eigenvalues.dat";

        // set up input
        viennacl::linalg::InputData input(diagonal, superdiagonal, mat_size);
        // compute Gerschgorin interval
        float lg = FLT_MAX;
        float ug = -FLT_MAX;
        //computeGerschgorin(input.a, input.b + 1, mat_size, lg, ug);
        computeGerschgorin(input.std_a, input.std_b, mat_size, lg, ug);
        printf("Gerschgorin interval: %f / %f\n", lg, ug);

        if (mat_size <= MAX_SMALL_MATRIX)
        {
          // initialize memory for result
          viennacl::linalg::ResultDataSmall result(mat_size);

          // run the kernel
          viennacl::linalg::computeEigenvaluesSmallMatrix(input, result, mat_size, lg, ug,
                                        precision);

          // get the result from the device and do some sanity checks,
          // save the result
          viennacl::linalg::processResultSmallMatrix(input, result, mat_size, result_file);
          eigenvalues = result.std_eigenvalues;
          bCompareResult = true;
        }

        else
        {
          // initialize memory for result
          viennacl::linalg::ResultDataLarge result(mat_size);

          // run the kernel
          viennacl::linalg::computeEigenvaluesLargeMatrix(input, result, mat_size,
                                        lg, ug, precision);


           // get the result from the device and do some sanity checks
          // save the result if user specified matrix size
          bCompareResult = viennacl::linalg::processResultDataLargeMatrix(input, result, mat_size, result_file);

          eigenvalues = result.std_eigenvalues;
        } //Large end
        return bCompareResult;
    }
  }
}
#endif
