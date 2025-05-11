/**
 * @file radauthreetimesteppingode.cc
 * @brief NPDE homework RadauThreeTimestepping
 * @author Erick Schulz
 * @date 08/04/2019
 * @copyright Developed at ETH Zurich
 */

#include "radauthreetimesteppingode.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace RadauThreeTimestepping {

/* SAM_LISTING_BEGIN_1 */
std::vector<double> twoStageRadauTimesteppingLinScalODE(unsigned int m) {
  std::vector<double> sol_vec;
  //====================
  double y0 = 1.0;

  double tau = 5.0/m;
//  sol_vec.push_back(y0);
  double evolution_op =  (1.0 - tau*(1.0 + tau/6.0)/((1.0 + tau*5.0/12.0)*(1.0 + tau/4.0) + tau*tau/16.0));

  for(int i = 0; i <= m; ++i){
    sol_vec.push_back(y0);
    y0 = evolution_op * y0;

  }


  //====================
  return sol_vec;
}
/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_2 */
void testConvergenceTwoStageRadauLinScalODE() {
  constexpr int nIter = 10;       // total number of iterations
  double max_norm_errors[nIter];  // errors vector for all approx. sols
  double rates[nIter - 1];        // The rates of convergence
  double avg_rate = 0.0;  // The average rate of convergence over all iterations

  //====================

  for(int k = 0; k < nIter; ++k){
    std::vector<double> exact_sol;
    int m = 10 * std::pow(2,k);
    double tau = 5.0/m;
    for(int i = 0; i <= m; ++i){
      exact_sol.push_back(std::exp(-i*tau));
    }
    std::vector<double>sol =  twoStageRadauTimesteppingLinScalODE(m);
    double max = 0;
    for(int i = 0; i <= m; ++i){
      if(std::abs(exact_sol[i] - sol[i]) > max) max = std::abs(exact_sol[i] - sol[i]);
    }
    max_norm_errors[k] = max;

  }
  for(int k = 0; k < nIter - 1; ++k){
    rates[k] = log2(max_norm_errors[k]/max_norm_errors[k+1]);
    avg_rate += rates[k];
  }
  avg_rate /= nIter - 1;


  //====================
  /* SAM_LISTING_END_2 */

  // Printing results
  std::cout << "\n" << std::endl;
  std::cout << "*********************************************************"
            << std::endl;
  std::cout << "         Convergence of two-stage Radau Method           "
            << std::endl;
  std::cout << "*********************************************************"
            << std::endl;
  std::cout << "--------------------- RESULTS ---------------------------"
            << std::endl;
  std::cout << "Iteration"
            << "\t| Nsteps"
            << "\t| error"
            << "\t\t| rates" << std::endl;
  std::cout << "---------------------------------------------------------"
            << std::endl;
  for (int k = 0; k < nIter; k++) {
    std::cout << k << "\t"
              << "\t|" << 10 * std::pow(2, k) << "\t\t|" << max_norm_errors[k];
    if (k > 0) {
      std::cout << "\t|" << rates[k - 1];
    }
    std::cout << "\n";
  }
  std::cout << "---------------------------------------------------------"
            << std::endl;
  std::cout << "Average rate of convergence: " << avg_rate << "\n" << std::endl;
}

}  // namespace RadauThreeTimestepping
