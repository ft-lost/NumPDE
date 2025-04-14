/**
 * @file
 * @brief NPDE homework NonConformingCrouzeixRaviartFiniteElements code
 * @author Anian Ruoss, edited Amélie Loher
 * @date   18.03.2019, 03.03.20
 * @copyright Developed at ETH Zurich
 */

#ifndef NUMPDE_L2_ERROR_CR_DISCRETIZATION_DIRICHLET_BVP_H
#define NUMPDE_L2_ERROR_CR_DISCRETIZATION_DIRICHLET_BVP_H

#include <lf/io/io.h>
#include <lf/mesh/mesh.h>

#include <cmath>
#include <string>

#include "crdirichletbvp.h"
#include "crfespace.h"
#include "crl2error.h"

namespace NonConformingCrouzeixRaviartFiniteElements {

/* SAM_LISTING_BEGIN_1 */
double L2errorCRDiscretizationDirichletBVP(const std::string &filename) {
  double l2_error;

// TODO: task 2-14.x)
  //====================
  auto factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(factory), filename);

  auto mesh_ptr = reader.mesh();

  auto f = [&] (Eigen::Vector2d x){
    double result = (2*M_PI*M_PI + x(0)*x(1))*std::sin(M_PI*x(0))*std::sin(M_PI*x(1));
    return result;
  };
  auto c = [&] (Eigen::Vector2d x){
    double result = x(0)*x(1);
    return result;
  };
  auto u = [&] (Eigen::Vector2d x){
    double result = std::sin(M_PI*x(0))*std::sin(M_PI*x(1));
    return result;
  };

  auto fe_space = std::make_shared<CRFeSpace>(reader.mesh());

  Eigen::VectorXd mu = solveCRDirichletBVP(fe_space, c, f);

  l2_error = computeCRL2Error(fe_space, mu, u);
  //====================
  return l2_error;
}
/* SAM_LISTING_END_1 */

}  // namespace NonConformingCrouzeixRaviartFiniteElements

#endif  // NUMPDE_L2_ERROR_CR_DISCRETIZATION_DIRICHLET_BVP_H
