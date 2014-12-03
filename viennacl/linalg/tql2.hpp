#ifndef VIENNACL_LINALG_TQL2_HPP
#define VIENNACL_LINALG_TQL2_HPP

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

/** @file viennacl/linalg/tql2.hpp
    @brief Implementation of the tql2-algorithm for eigenvalue computations.
*/


#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include <iomanip>

#include "viennacl/linalg/qr-method-common.hpp"
#include "viennacl/linalg/prod.hpp"

namespace viennacl
{
namespace linalg
{
  // Symmetric tridiagonal QL algorithm.
  // This is derived from the Algol procedures tql1, by Bowdler, Martin, Reinsch, and Wilkinson,
  // Handbook for Auto. Comp., Vol.ii-Linear Algebra, and the corresponding Fortran subroutine in EISPACK.
  template <typename SCALARTYPE, typename VectorType>
  void tql1(int n,
            VectorType & d,
            VectorType & e)
  {
      for (int i = 1; i < n; i++)
          e[i - 1] = e[i];


      e[n - 1] = 0;

      SCALARTYPE f = 0;
      SCALARTYPE tst1 = 0;
      SCALARTYPE eps = 1e-6;
     // SCALARTYPE eps = static_cast<SCALARTYPE>(viennacl::linalg::detail::EPS);
     //  std::cout << "eps = " << eps << std::endl;

      for (int l = 0; l < n; l++)
      {
          // Find small subdiagonal element.
          tst1 = std::max<SCALARTYPE>(tst1, std::fabs(d[l]) + std::fabs(e[l]));
          int m = l;
          while (m < n)
          {
              if (std::fabs(e[m]) <= eps * tst1)
                  break;
              m++;
          }

          // If m == l, d[l) is an eigenvalue, otherwise, iterate.
          if (m > l)
          {
              int iter = 0;
              do
              {
                  iter = iter + 1;  // (Could check iteration count here.)

                  // Compute implicit shift
                  SCALARTYPE g = d[l];
                  SCALARTYPE p = (d[l + 1] - g) / (2 * e[l]);
                  SCALARTYPE r = viennacl::linalg::detail::pythag<SCALARTYPE>(p, 1);
                  if (p < 0)
                  {
                      r = -r;
                  }

                  d[l] = e[l] / (p + r);
                  d[l + 1] = e[l] * (p + r);
                  SCALARTYPE h = g - d[l];
                  for (int i = l + 2; i < n; i++)
                  {
                      d[i] -= h;
                  }

                  f = f + h;

                  // Implicit QL transformation.
                  p = d[m];
                  SCALARTYPE c = 1;
                  SCALARTYPE s = 0;
                  for (int i = m - 1; i >= l; i--)
                  {
                      g = c * e[i];
                      h = c * p;
                      r = viennacl::linalg::detail::pythag(p, e[i]);
                      e[i + 1] = s * r;
                      s = e[i] / r;
                      c = p / r;
                      p = c * d[i] - s * g;
                      d[i + 1] = h + s * (c * g + s * d[i]);
                  }
                  e[l] = s * p;
                  d[l] = c * p;
              // Check for convergence.
              }
              while (std::fabs(e[l]) > eps * tst1);
          }
          d[l] = d[l] + f;
          e[l] = 0;
      }
  }







// Symmetric tridiagonal QL algorithm.
// This is derived from the Algol procedures tql2, by Bowdler, Martin, Reinsch, and Wilkinson,
// Handbook for Auto. Comp., Vol.ii-Linear Algebra, and the corresponding Fortran subroutine in EISPACK.
template <typename SCALARTYPE, typename VectorType, typename F>
void tql2(matrix_base<SCALARTYPE, F> & Q,
          VectorType & d,
          VectorType & e)
{
    int n = static_cast<int>(viennacl::traits::size1(Q));

    //boost::numeric::ublas::vector<SCALARTYPE> cs(n), ss(n);
    std::vector<SCALARTYPE> cs(n), ss(n);
    viennacl::vector<SCALARTYPE> tmp1(n), tmp2(n);

    for (int i = 1; i < n; i++)
        e[i - 1] = e[i];

    e[n - 1] = 0;

    SCALARTYPE f = 0;
    SCALARTYPE tst1 = 0;
    SCALARTYPE eps = static_cast<SCALARTYPE>(viennacl::linalg::detail::EPS);

    for (int l = 0; l < n; l++)
    {
        // Find small subdiagonal element.
        tst1 = std::max<SCALARTYPE>(tst1, std::fabs(d[l]) + std::fabs(e[l]));
        int m = l;
        while (m < n)
        {
            if (std::fabs(e[m]) <= eps * tst1)
                break;
            m++;
        }

        // If m == l, d[l) is an eigenvalue, otherwise, iterate.
        if (m > l)
        {
            int iter = 0;
            do
            {
                iter = iter + 1;  // (Could check iteration count here.)

                // Compute implicit shift
                SCALARTYPE g = d[l];
                SCALARTYPE p = (d[l + 1] - g) / (2 * e[l]);
                SCALARTYPE r = viennacl::linalg::detail::pythag<SCALARTYPE>(p, 1);
                if (p < 0)
                {
                    r = -r;
                }

                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                SCALARTYPE dl1 = d[l + 1];
                SCALARTYPE h = g - d[l];
                for (int i = l + 2; i < n; i++)
                {
                    d[i] -= h;
                }

                f = f + h;

                // Implicit QL transformation.
                p = d[m];
                SCALARTYPE c = 1;
                SCALARTYPE c2 = c;
                SCALARTYPE c3 = c;
                SCALARTYPE el1 = e[l + 1];
                SCALARTYPE s = 0;
                SCALARTYPE s2 = 0;
                for (int i = m - 1; i >= l; i--)
                {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = viennacl::linalg::detail::pythag(p, e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);


                    cs[i] = c;
                    ss[i] = s;
                }


                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;

                viennacl::copy(cs, tmp1);
                viennacl::copy(ss, tmp2);

                viennacl::linalg::givens_next(Q, tmp1, tmp2, l, m);

                // Check for convergence.
            }
            while (std::fabs(e[l]) > eps * tst1);
        }
        d[l] = d[l] + f;
        e[l] = 0;
    }

    // Sort eigenvalues and corresponding vectors.
/*
       for (int i = 0; i < n-1; i++) {
          int k = i;
          SCALARTYPE p = d[i];
          for (int j = i+1; j < n; j++) {
             if (d[j] > p) {
                k = j;
                p = d[j);
             }
          }
          if (k != i) {
             d[k] = d[i];
             d[i] = p;
             for (int j = 0; j < n; j++) {
                p = Q(j, i);
                Q(j, i) = Q(j, k);
                Q(j, k) = p;
             }
          }
       }

*/

}
} // namespace linalg
} // namespace viennacl
#endif
