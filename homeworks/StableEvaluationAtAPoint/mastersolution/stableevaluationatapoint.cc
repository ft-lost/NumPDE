/**
 * @file stableevaluationatapoint.cc
 * @brief NPDE homework StableEvaluationAtAPoint
 * @author Amélie Loher, Erick Schulz & Philippe Peter
 * @date 29.11.2021
 * @copyright Developed at ETH Zurich
 */

#include "stableevaluationatapoint.h"

#include <lf/base/base.h>
#include <lf/fe/fe.h>
#include <lf/geometry/geometry.h>
#include <lf/mesh/mesh.h>
#include <lf/quad/quad.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

namespace StableEvaluationAtAPoint {

double MeshSize(const std::shared_ptr<const lf::mesh::Mesh> &mesh_p) {
  double mesh_size = 0.0;
  // Find maximal edge length
  for (const lf::mesh::Entity *edge : mesh_p->Entities(1)) {
    // Compute the length of the edge
    double edge_length = lf::geometry::Volume(*(edge->Geometry()));
    mesh_size = std::max(edge_length, mesh_size);
  }
  return mesh_size;
}

Eigen::Vector2d OuterNormalUnitSquare(Eigen::Vector2d x) {
  // Use shortcut: x is on the unit square
  if (x(0) > x(1) && x(0) < 1.0 - x(1)) {
    return Eigen::Vector2d(0.0, -1.0);
  }
  if (x(0) > x(1) && x(0) > 1.0 - x(1)) {
    return Eigen::Vector2d(1.0, 0.0);
  }
  if (x(0) < x(1) && x(0) > 1.0 - x(1)) {
    return Eigen::Vector2d(0.0, 1.0);
  }
  return Eigen::Vector2d(-1.0, 0.0);
}

double FundamentalSolution::operator()(Eigen::Vector2d y) {
  LF_ASSERT_MSG(x_ != y, "G not defined for these coordinates!");
  return -1.0 / (2.0 * M_PI) * std::log((x_ - y).norm());
}

Eigen::Vector2d FundamentalSolution::grad(Eigen::Vector2d y) {
  LF_ASSERT_MSG(x_ != y, "G not defined for these coordinates!");
  return (x_ - y) / (2.0 * M_PI * (x_ - y).squaredNorm());
}

double PointEval(std::shared_ptr<const lf::mesh::Mesh> mesh_p) {
  double error = 0.0;
  const auto u = [](Eigen::Vector2d x) -> double {
    Eigen::Vector2d one(1.0, 0.0);
    return std::log((x + one).norm());
  };
  const auto gradu = [](Eigen::Vector2d x) -> Eigen::Vector2d {
    Eigen::Vector2d one(1.0, 0.0);
    return (x + one) / (x + one).squaredNorm();
  };
  // Define a Functor for the dot product of grad u(x) * n(x)
  const auto gradu_dot_n = [gradu](const Eigen::Vector2d x) -> double {
    // Determine the normal vector n on the unit square
    Eigen::Vector2d n = OuterNormalUnitSquare(x);
    return gradu(x).dot(n);
  };

  // Compute right hand side
  const Eigen::Vector2d x(0.3, 0.4);
  const double rhs = PSL(mesh_p, gradu_dot_n, x) - PDL(mesh_p, u, x);
  // Compute the error
  error = std::abs(u(x) - rhs);
  return error;
}

double Psi::operator()(Eigen::Vector2d y) {
  const double c = M_PI / (0.5 * std::sqrt(2) - 1.0);
  const double dist = (y - center_).norm();

  if (dist <= 0.25 * std::sqrt(2)) {
    return 0.0;
  }
  if (dist >= 0.5) {
    return 1.0;
  }
  return std::pow(std::cos(c * (dist - 0.5)), 2);
}

Eigen::Vector2d Psi::grad(Eigen::Vector2d y) {
  double c = M_PI / (0.5 * std::sqrt(2) - 1.0);
  double dist = (y - center_).norm();

  if (dist <= 0.25 * std::sqrt(2)) {
    return Eigen::Vector2d(0.0, 0.0);
  }
  if (dist >= 0.5) {
    return Eigen::Vector2d(0.0, 0.0);
  }
  return -2.0 * std::cos(c * (dist - 0.5)) * std::sin(c * (dist - 0.5)) *
         (c / dist) * (y - center_);
}

double Psi::lapl(Eigen::Vector2d y) {
  double c = M_PI / (0.5 * std::sqrt(2) - 1.0);
  double c2 = c * c;
  double dist = (y - center_).norm();

  if (dist <= 0.25 * std::sqrt(2)) {
    return 0.0;
  }
  if (dist >= 0.5) {
    return 0.0;
  }
  double sineval = std::sin(c * (dist - 0.5));
  double coseval = std::cos(c * (dist - 0.5));
  return 2 * c2 * sineval * sineval - 2 * c2 * coseval * coseval -
         2 * c * sineval * coseval / dist;
}

double Jstar(std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
             Eigen::VectorXd uFE, const Eigen::Vector2d x) {
  double val = 0.0;
  Psi psi(Eigen::Vector2d(0.5, 0.5));
  FundamentalSolution G(x);

  // Mesh covering a unit square domain
  std::shared_ptr<const lf::mesh::Mesh> mesh = fe_space->Mesh();
  // Use midpoint quadrature rule
  const lf::quad::QuadRule qr = lf::quad::make_TriaQR_MidpointRule();
  // Quadrature points
  const Eigen::MatrixXd zeta_ref{qr.Points()};
  // Quadrature weights
  const Eigen::VectorXd w_ref{qr.Weights()};
  // Number of quadrature points
  const lf::base::size_type P = qr.NumPoints();
  // Create mesh function to be evaluated at the quadrature points
  auto uFE_mf = lf::fe::MeshFunctionFE(fe_space, uFE);

  // Loop over all cells
  for (const lf::mesh::Entity *entity : mesh->Entities(0)) {
    // Standard way to apply a local quadrature rule
    const lf::geometry::Geometry &geo{*entity->Geometry()};
    // Quadrature points on actual cell
    const Eigen::MatrixXd zeta{geo.Global(zeta_ref)};
    const Eigen::VectorXd gram_dets{geo.IntegrationElement(zeta_ref)};
    // Values of finite element function on all quadrature points
    auto u_vals = uFE_mf(*entity, zeta_ref);

    // Quadrature loop
    for (int l = 0; l < P; l++) {
      const double w = w_ref[l] * gram_dets[l];
      val += w * (-u_vals[l]) *
             (2.0 * (G.grad(zeta.col(l))).dot(psi.grad(zeta.col(l))) +
              G(zeta.col(l)) * psi.lapl(zeta.col(l)));
    }
  }
  return val;
}

double StablePointEvaluation(
    std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
    Eigen::VectorXd uFE, const Eigen::Vector2d x) {
  double res = 0.0;

  Eigen::Vector2d center(0.5, 0.5);
  if ((x - center).norm() <= 0.25) {
    res = Jstar(fe_space, uFE, x);
  } else {
    std::cerr << "The point does not fulfill the assumptions" << std::endl;
  }

  return res;
}

double EvaluateFEFunction(
    std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
    const Eigen::VectorXd &uFE, Eigen::Vector2d global, double tol) {
  // Extract mesh
  auto mesh_p = fe_space->Mesh();
  // wrap coefficient vector into a FE mesh-function
  lf::fe::MeshFunctionFE mf(fe_space, uFE);

  for (const lf::mesh::Entity *entity_p : mesh_p->Entities(0)) {
    LF_ASSERT_MSG(lf::base::RefEl::kTria() == entity_p->RefEl(),
                  "Function only defined for triangular cells");

    // compute geometric information about the cell
    const lf::geometry::Geometry *geo_p = entity_p->Geometry();
    Eigen::MatrixXd corners = lf::geometry::Corners(*geo_p);

    // transform global coordinates to local coordinates on the cell
    Eigen::Matrix2d A;
    A << corners.col(1) - corners.col(0), corners.col(2) - corners.col(0);
    Eigen::Vector2d b;
    b << global - corners.col(0);
    Eigen::Vector2d loc = A.fullPivLu().solve(b);

    // evaluate meshfunction, if local coordinates lie in the reference triangle
    if (loc(0) >= 0 - tol && loc(1) >= 0 - tol && loc(0) + loc(1) <= 1 + tol) {
      return mf(*entity_p, loc)[0];
    }
  }
  return 0.0;
}

}  // namespace StableEvaluationAtAPoint