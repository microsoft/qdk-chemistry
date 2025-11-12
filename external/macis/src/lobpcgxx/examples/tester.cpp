#include <iomanip>
#include <iostream>
#include <lobpcgxx/lobpcg.hpp>
#include <memory>
#include <random>
#include <vector>

template <typename T>
int test() {
  const int64_t n = 1000;
  const int64_t k = 20;

  // matrix A -> 2D Laplacian
  std::vector<T> A(n * n, 0.);
  A[0] = 2.0;
  for (int64_t i = 1; i < n; ++i) {
    A[(i - 1) * n + i] = -1.0;
    A[i * n + i] = 2.0;
    A[i * n + (i - 1)] = -1.0;
  }

  // Aop
  lobpcgxx::operator_action_type<T> Aop = [&](int64_t n, int64_t k, const T* x,
                                              int64_t ldx, T* y,
                                              int64_t ldy) -> void {
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, n,
               k, n, T(1.0), A.data(), n, x, ldx, T(0.0), y, ldy);
  };

  // starting vectors
  std::vector<T> X(n * k);

  // Structured vectors
  // for ( int64_t i = 0; i < n/2; ++i ) {
  //  X[2*i] = 1.0 / std::sqrt(n/2) ;
  //  X[2*i+n+1] = 1.0 / std::sqrt(n/2) ;
  //}

  // Random vectors
  std::default_random_engine gen;
  std::normal_distribution<> dist(0., 1.);
  auto rand_gen = [&]() { return dist(gen); };
  std::generate(X.begin(), X.end(), rand_gen);
  lobpcgxx::cholqr(n, k, X.data(), n);  // Orthogonalize

  lobpcgxx::lobpcg_settings settings;
  settings.conv_tol = 1e-6;
  settings.maxiter = 2000;
  lobpcgxx::lobpcg_operator<T> lob_op(Aop);

  std::vector<T> lam(k), res(k);
  lobpcgxx::lobpcg(settings, n, k, k, lob_op, lam.data(), X.data(), n,
                   res.data());

  std::cout << std::scientific << std::setprecision(10) << std::endl;
  for (int64_t i = 0; i < k; ++i) {
    std::cout << "  lam[" << i << "] = " << lam[i] << ", res[" << i
              << "] = " << res[i] << std::endl;
  }

  return 0;
}

int main() {
  // double
  std::cout << "\n+++ TEST for double +++\n\n";
  test<double>();

  // float
  // std::cout << "\n+++ TEST for float +++\n\n" ;
  // test< float >() ;

  // successful
  std::cout << ">> end test lobpcg <<" << std::endl;

  return 0;
}
