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


#include "viennacl/linalg/detail/bisect/structs.hpp"
#include "viennacl/linalg/detail/bisect/gerschgorin.hpp"
#include "viennacl/linalg/detail/bisect/bisect_large.hpp"
#include "viennacl/linalg/detail/bisect/bisect_small.hpp"


namespace viennacl
{
  namespace linalg
  {
    template<typename NumericT>
    bool
    bisect(const std::vector<NumericT> & diagonal, const std::vector<NumericT> & superdiagonal, std::vector<NumericT> & eigenvalues)
    {
      assert(diagonal.size() == superdiagonal.size() &&
             diagonal.size() == eigenvalues.size()   &&
             bool("Input vectors do not have the same sizes!"));
      bool bCompareResult = false;
      // flag if the matrix size is due to explicit user request
      // desired precision of eigenvalues
      NumericT  precision = 0.00001;
      const unsigned int mat_size = diagonal.size();

      // set up input
      viennacl::linalg::detail::InputData<NumericT> input(diagonal, superdiagonal, mat_size);

      NumericT lg =  FLT_MAX;
      NumericT ug = -FLT_MAX;
      // compute Gerschgorin interval
      viennacl::linalg::detail::computeGerschgorin(input.std_a, input.std_b, mat_size, lg, ug);

      // decide wheter the algorithm for small or for large matrices will be started
      if (mat_size <= MAX_SMALL_MATRIX)
      {
        // initialize memory for result
        viennacl::linalg::detail::ResultDataSmall<NumericT> result(mat_size);

        // run the kernel
        viennacl::linalg::detail::computeEigenvaluesSmallMatrix(input, result, mat_size, lg, ug,
                                      precision);

        // get the result from the device and do some sanity checks,
        viennacl::linalg::detail::processResultSmallMatrix(result, mat_size);
        eigenvalues = result.std_eigenvalues;
        bCompareResult = true;
      }

      else
      {
        // initialize memory for result
        viennacl::linalg::detail::ResultDataLarge<NumericT> result(mat_size);

        // run the kernel
        viennacl::linalg::detail::computeEigenvaluesLargeMatrix(input, result, mat_size,
                                      lg, ug, precision);

        // get the result from the device and do some sanity checks
        bCompareResult = viennacl::linalg::detail::processResultDataLargeMatrix(result, mat_size);

        eigenvalues = result.std_eigenvalues;
      }
      return bCompareResult;
    }


    template<typename NumericT>
    bool
    bisect(const viennacl::vector<NumericT> & diagonal, const viennacl::vector<NumericT> & superdiagonal, viennacl::vector<NumericT> & eigenvalues)
    {
      assert(diagonal.size() == superdiagonal.size() &&
             diagonal.size() == eigenvalues.size()   &&
             bool("Input vectors do not have the same sizes!"));
      bool bCompareResult = false;
      // flag if the matrix size is due to explicit user request
      // desired precision of eigenvalues
      NumericT  precision = 0.00001;
      const unsigned int mat_size = diagonal.size();

      // set up input
      viennacl::linalg::detail::InputData<NumericT> input(diagonal, superdiagonal, mat_size);

      NumericT lg =  FLT_MAX;
      NumericT ug = -FLT_MAX;
      // compute Gerschgorin interval
      viennacl::linalg::detail::computeGerschgorin(input.std_a, input.std_b, mat_size, lg, ug);

      // decide wheter the algorithm for small or for large matrices will be started
      if (mat_size <= MAX_SMALL_MATRIX)
      {
        // initialize memory for result
        viennacl::linalg::detail::ResultDataSmall<NumericT> result(mat_size);

        // run the kernel
        viennacl::linalg::detail::computeEigenvaluesSmallMatrix(input, result, mat_size, lg, ug,
                                      precision);

        // get the result from the device and do some sanity checks,
        viennacl::linalg::detail::processResultSmallMatrix(result, mat_size);
        copy(result.std_eigenvalues, eigenvalues);
        bCompareResult = true;
      }

      else
      {
        // initialize memory for result
        viennacl::linalg::detail::ResultDataLarge<NumericT> result(mat_size);

        // run the kernel
        viennacl::linalg::detail::computeEigenvaluesLargeMatrix(input, result, mat_size,
                                      lg, ug, precision);

        // get the result from the device and do some sanity checks
        bCompareResult = viennacl::linalg::detail::processResultDataLargeMatrix(result, mat_size);

        copy(result.std_eigenvalues, eigenvalues);
      }
      return bCompareResult;
    }
  }
}
#endif
