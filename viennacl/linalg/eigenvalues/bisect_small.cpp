/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Computation of eigenvalues of a small bidiagonal matrix */

#ifndef _BISECT_SMALL_CUH_
#define _BISECT_SMALL_CUH_

//namespace viennacl
//{
  //namespace linalg
  //{

    extern "C" {

        ////////////////////////////////////////////////////////////////////////////////
        //! Determine eigenvalues for matrices smaller than MAX_SMALL_MATRIX
        //! @param TimingIterations  number of iterations for timing
        //! @param  input  handles to input data of kernel
        //! @param  result handles to result of kernel
        //! @param  mat_size  matrix size
        //! @param  lg  lower limit of Gerschgorin interval
        //! @param  ug  upper limit of Gerschgorin interval
        //! @param  precision  desired precision of eigenvalues
        //! @param  iterations  number of iterations for timing
        ////////////////////////////////////////////////////////////////////////////////

void generate_computeEigenvaluesSmallMatrix(StringType & source, std::string const & numeric_string)
{
    source.append("         void  \n");
    source.append("         computeEigenvaluesSmallMatrix(const InputData &input, ResultDataSmall &result,  \n");
    source.append("                                       const unsigned int mat_size,  \n");
    source.append("                                       const float lg, const float ug,  \n");
    source.append("                                       const float precision);  \n");


} 


////////////////////////////////////////////////////////////////////////////////
//! Initialize variables and memory for the result for small matrices
//! @param result  handles to the necessary memory
//! @param  mat_size  matrix_size
////////////////////////////////////////////////////////////////////////////////

void generate_initResultSmallMatrix(StringType & source, std::string const & numeric_string)
{
    source.append("         void  \n");
    source.append("         initResultSmallMatrix(ResultDataSmall &result, const unsigned int mat_size);  \n");


} 


////////////////////////////////////////////////////////////////////////////////
//! Cleanup memory and variables for result for small matrices
//! @param  result  handle to variables
////////////////////////////////////////////////////////////////////////////////

void generate_cleanupResultSmallMatrix(StringType & source, std::string const & numeric_string)
{
    source.append("         void  \n");
    source.append("         cleanupResultSmallMatrix(ResultDataSmall &result);  \n");

        ////////////////////////////////////////////////////////////////////////////////
        //! Process the result obtained on the device, that is transfer to host and
        //! perform basic sanity checking
        //! @param  input   handles to input data
        //! @param  result  handles to result variables
        //! @param  mat_size   matrix size
        //! @param  filename  output filename
        ////////////////////////////////////////////////////////////////////////////////
} 

////////////////////////////////////////////////////////////////////////////////
//! Process the result obtained on the device, that is transfer to host and
//! perform basic sanity checking
//! @param  input   handles to input data
//! @param  result  handles to result variables
//! @param  mat_size   matrix size
//! @param  filename  output filename
////////////////////////////////////////////////////////////////////////////////



void generate_processResultSmallMatrix(StringType & source, std::string const & numeric_string)
{
    source.append("         void  \n");
    source.append("         processResultSmallMatrix(const InputData &input, ResultDataSmall &result,  \n");
    source.append("                                  const unsigned int mat_size, const char *filename);  \n");

}
    }
  //}
//}
#endif // #ifndef _BISECT_SMALL_CUH_



 
