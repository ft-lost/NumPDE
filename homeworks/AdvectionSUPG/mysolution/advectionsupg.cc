/**
 * @file advectionsupg.cc
 * @brief NPDE homework AdvectionSUPG code
 * @author R. Hiptmair
 * @date July 2022
 * @copyright Developed at SAM, ETH Zurich
 */

#include "advectionsupg.h"

#include <lf/base/lf_assert.h>
#include <lf/base/ref_el.h>
#include <lf/mesh/hybrid2d/mesh.h>
#include <lf/mesh/hybrid2d/mesh_factory.h>
#include <lf/refinement/mesh_hierarchy.h>
#include <lf/uscalfe/lagr_fe.h>
#include <math.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "lf/mesh/test_utils/test_meshes.h"

namespace AdvectionSUPG {

/* SAM_LISTING_BEGIN_9 */
void enforce_boundary_conditions(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fe_space,
    lf::assemble::COOMatrix<double>& A, Eigen::VectorXd& b) {
  const lf::mesh::utils::MeshFunctionGlobal mf_g_Gamma_in{
      [](const Eigen::Vector2d& x) { return std::pow(sin(M_PI * x(0)), 2); }};

  lf::mesh::utils::AllCodimMeshDataSet<bool> bd_flags(fe_space->Mesh(), false);
  // Hint: Fill bd_flags
  // ========================================
  // Your code here
  // ========================================
  auto flag_values{lf::fe::InitEssentialConditionFromFunction(
      *fe_space, bd_flags, mf_g_Gamma_in)};

  lf::assemble::FixFlaggedSolutionCompAlt<double>(
      [&flag_values](lf::assemble::glb_idx_t dof_idx) {
        return flag_values[dof_idx];
      },
      A, b);
};
/* SAM_LISTING_END_9 */

/* SAM_LISTING_BEGIN_1 */
Eigen::VectorXd solveRotSUPG(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fe_space) {
  // ========================================
  // Your code here
  // ========================================
  return Eigen::VectorXd::Zero(fe_space->LocGlobMap().NumDofs());
};
/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_2 */
void cvgL2SUPG() {
  // ========================================
  // Your code here
  // ========================================
};
/* SAM_LISTING_END_2 */

/* SAM_LISTING_BEGIN_8 */
void visSolution(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fe_space,
    Eigen::VectorXd& u) {
  const lf::fe::MeshFunctionFE mf_sol(fe_space, u);
  lf::io::VtkWriter vtk_writer(fe_space->Mesh(), "./solution.vtk");
  vtk_writer.WritePointData("solution", mf_sol);
}
/* SAM_LISTING_END_8 */

}  // namespace AdvectionSUPG
