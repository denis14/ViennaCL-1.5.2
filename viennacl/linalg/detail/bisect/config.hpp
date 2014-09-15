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

/* Global configuration parameter */
#ifndef VIENNACL_LINALG_DETAIL_CONFIG_HPP_
#define VIENNACL_LINALG_DETAIL_CONFIG_HPP_

// should be power of two
#define  MAX_THREADS_BLOCK                256

#define  MAX_SMALL_MATRIX                 512
#define  MAX_THREADS_BLOCK_SMALL_MATRIX   512

#define  MIN_ABS_INTERVAL                 5.0e-37

 #endif // #ifndef VIENNACL_LINALG_DETAIL_CONFIG_HPP_
