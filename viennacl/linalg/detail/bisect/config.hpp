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

/* Global configuration parameter */
#ifndef VIENNACL_LINALG_DETAIL_CONFIG_HPP_
#define VIENNACL_LINALG_DETAIL_CONFIG_HPP_

// should be power of two
#define  MAX_THREADS_BLOCK                256

#define  MAX_SMALL_MATRIX                 512
#define  MAX_THREADS_BLOCK_SMALL_MATRIX   512

#define  MIN_ABS_INTERVAL                 5.0e-37

 #endif // #ifndef VIENNACL_LINALG_DETAIL_CONFIG_HPP_
