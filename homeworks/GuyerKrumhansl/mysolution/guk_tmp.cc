/**
 * @file guyerkrumhansl.cc
 * @brief NPDE homework GuyerKrumhansl code
 * @author R. Hiptmair
 * @date July 2023
 * @copyright Developed at SAM, ETH Zurich
 */

#include <iostream>
#include <utility>

#include "guyerkrumhansl.h"

namespace GuyerKrumhansl {

/* Demo code: efficient direct initialization of a sparse matrix */
/* SAM_LISTING_BEGIN_1 */
void demo_EigenSparseInit(unsigned int n = 20, unsigned int m = 10) {
  // Set up zero sparse matrix with row major storage format
  // This format is essential for being able to set the maximal
  // number of non-zero entries \textbf{per row}.
  Eigen::SparseMatrix<double, Eigen::RowMajor> X(n, m);
  // Reserve space for at most nnz\_row non-zero entries per row
  const std::size_t nnz_row = 3;
  X.reserve(Eigen::VectorXi::Constant(n, nnz_row));
  // Initialize nnz\_row  entries per row
  for (int row_idx = 0; row_idx < n; ++row_idx) {
    for (int k = 0; k < nnz_row; ++k) {
      X.insert(row_idx, (row_idx * k) % m) = 1.0;
    }
  }
  Eigen::MatrixXd X_dense = X;
  std::cout << "Matrix X = " << std::endl << X_dense << std::endl;
}
/* SAM_LISTING_END_1 */

// Initialization of matrices for MOL ODE
/* SAM_LISTING_BEGIN_2 */
std::pair<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>>
generateMOLMatrices(unsigned int M, double rho, double sigma, double mu,
                    double kappa) {
  const unsigned int N = 2 * M - 1;
  const double h = 1.0 / M;
  Eigen::SparseMatrix<double> Mh(N, N);
  Eigen::SparseMatrix<double> Ah(N, N);
  Mh.reserve(Eigen::VectorXi::Constant(N, 3));
  Ah.reserve(Eigen::VectorXi::Constant(N, 5));
  // I. Matrix $\wh{\VM}$
  // Initialize diagonal $\VM$-part of $\wh{\VM}$
  for (int j = 0; j < M; ++j) {
    Mh.insert(j, j) = rho * h;
  }
  // Initialize tri-diagonal $\VN$-part of $\wh{\VM}$
  const double fac = sigma * h / 6.0;
  Mh.insert(M, M) = 4.0 * fac;
  Mh.insert(M, M + 1) = fac;
  for (int j = 1; j < M - 2; ++j) {
    Mh.insert(j + M, j + M) = 4 * fac;
    Mh.insert(j + M, j + M - 1) = fac;
    Mh.insert(j + M, j + M + 1) = fac;
  }
  Mh.insert(N - 1, N - 1) = 4 * fac;
  Mh.insert(N - 1, N - 2) = fac;
  // II. Matrix $\wh{\VA}$
  // Initialize bi-diagonal $\VB$-part and $\VC$-part of $\wh{\VA}$, see
  // \prbeqref{eq:Bmat}
  Ah.insert(0, M) = -1.0;
  Ah.insert(M, 0) = -kappa;
  for (int j = 1; j < M - 1; ++j) {
    Ah.insert(j, M + j) = -1.0;
    Ah.insert(j, M + j - 1) = 1.0;
    Ah.insert(M + j, j) = -kappa;
    Ah.insert(M + j - 1, j) = kappa;
  }
  Ah.insert(M - 1, N - 1) = 1.0;
  Ah.insert(N - 1, M - 1) = kappa;
  // Initialize tri-diagonal $\VD$-part of $\wh{\VA}$, see \prbeqref{m:D}
  const double diag = -2.0 * (h / 3.0 + mu / h);
  const double offd = -h / 6.0 + mu / h;
  Ah.insert(M, M) = diag;
  Ah.insert(M, M + 1) = offd;
  for (int j = 1; j < M - 1; ++j) {
    Ah.insert(M + j, M + j) = diag;
    Ah.insert(M + j, M + j - 1) = offd;
    Ah.insert(M + j, M + j + 1) = offd;
  }
  Ah.insert(N - 1, N - 1) = diag;
  Ah.insert(N - 1, N - 2) = offd;

  return {Mh, Ah};
}
/* SAM_LISTING_END_2 */

}  // namespace GuyerKrumhansl
