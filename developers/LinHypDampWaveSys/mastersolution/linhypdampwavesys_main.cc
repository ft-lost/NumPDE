/**
 * @ file linhypdampwavesys_main.cc
 * @ brief NPDE homework TEMPLATE MAIN FILE
 * @ author Wouter Tonnon
 * @ date May 2024
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>
#include <numeric>

#include "linhypdampwavesys.h"
#include "systemcall.h"

using namespace LinHypDampWaveSys;

int main(int /*argc*/, char** /*argv*/) {
  std::cout << "NumPDE homework problem LinHypDampWaveSys by W. Tonnon\n";

  // Undamped system
  double c = 1.;
  double r = 0.;
  double T = 2.;
  unsigned int N = 500;

  visWavSol(c, r, T, N, LF);
  systemcall::execute(
      "python3 scripts/vis_solution.py solution.csv solution_undamped_LF.eps");

  trackEnergysWavSol(c, r, T, N, LF);
  systemcall::execute(
      "python3 scripts/envis.py energies.csv energies_undamped_LF.eps");

  visWavSol(c, r, T, N, UW);
  systemcall::execute(
      "python3 scripts/vis_solution.py solution.csv solution_undamped_UW.eps");

  trackEnergysWavSol(c, r, T, N, UW);
  systemcall::execute(
      "python3 scripts/envis.py energies.csv energies_undamped_UW.eps");

  // Damped System
  c = 1.;
  r = 1.;
  T = 2.;
  N = 500;

  visWavSol(c, r, T, N, LF);
  systemcall::execute(
      "python3 scripts/vis_solution.py solution.csv solution_damped_LF.eps");

  trackEnergysWavSol(c, r, T, N, LF);
  systemcall::execute(
      "python3 scripts/envis.py energies.csv energies_damped_LF.eps");

  visWavSol(c, r, T, N, UW);
  systemcall::execute(
      "python3 scripts/vis_solution.py solution.csv solution_damped_UW.eps");

  trackEnergysWavSol(c, r, T, N, UW);
  systemcall::execute(
      "python3 scripts/envis.py energies.csv energies_damped_UW.eps");

  return 0;
}
