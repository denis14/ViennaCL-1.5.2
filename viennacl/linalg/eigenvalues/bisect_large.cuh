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

#ifndef _BISECT_LARGE_CUH_
#define _BISECT_LARGE_CUH_

extern "C" {

   
    ////////////////////////////////////////////////////////////////////////////////
    //! Initialize variables and memory for result
    //! @param  result handles to memory
    //! @param  matr_size  size of the matrix
    ////////////////////////////////////////////////////////////////////////////////
    void
    initResultDataLargeMatrix(ResultDataLarge &result, const unsigned int mat_size);

    ////////////////////////////////////////////////////////////////////////////////
    //! Cleanup result memory
    //! @param result  handles to memory
    ////////////////////////////////////////////////////////////////////////////////
    void
    cleanupResultDataLargeMatrix(ResultDataLarge &result);

    ////////////////////////////////////////////////////////////////////////////////
    //! Process the result, that is obtain result from device and do simple sanity
    //! checking
    //! @param  input   handles to input data
    //! @param  result  handles to result data
    //! @param  mat_size  matrix size
    //! @param  filename  output filename
    ////////////////////////////////////////////////////////////////////////////////
    bool
    processResultDataLargeMatrix(const InputData &input, ResultDataLarge &result,
                                 const unsigned int mat_size,
                                 const char *filename);

};

#endif // #ifndef _BISECT_LARGE_CUH_

