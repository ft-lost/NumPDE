/**
 * @file stokesminielement.cc
 * @brief NPDE homework StokesMINIElement code
 * @author Ralf Hiptmair
 * @date June 2024
 * @copyright Developed at SAM, ETH Zurich
 */

#include "stokesminielement.h"

#include <lf/mesh/test_utils/test_meshes.h>

namespace StokesMINIElement {

/* SAM_LISTING_BEGIN_1 */
SimpleFEMElementMatrixProvider::ElemMat SimpleFEMElementMatrixProvider::Eval(
    const lf::mesh::Entity& cell) {
  LF_VERIFY_MSG(cell.RefEl() == lf::base::RefEl::kTria(),
                "Unsupported cell type " << cell.RefEl());
  // Area of the triangle
  double area = lf::geometry::Volume(*cell.Geometry());
  // Compute gradients of barycentric coordinate functions, see
  // \lref{cpp:gradbarycoords}. First fetch vertices of the triangle
  auto endpoints = lf::geometry::Corners(*(cell.Geometry()));
  Eigen::Matrix<double, 3, 3> X;  // temporary matrix
  X.block<3, 1>(0, 0) = Eigen::Vector3d::Ones();
  X.block<3, 2>(0, 1) = endpoints.transpose();
  // This matrix contains $\cob{\grad \lambda_i}$ in its columns
  const auto G{X.inverse().block<2, 3>(1, 0)};
  // Inner products of gradients of barycentric coordinate functions
  // (Scaled) element matrix for $-\Delta$
  const Eigen::Matrix3d L{G.transpose() * G};
  // Fixed-size matrix containing the element matrix
  ElemMat AK;
  AK.setZero();
  // indices of local shape  function for various solution components
  constexpr std::array<Eigen::Index, 3> vx_idx_{0, 3, 6};
  constexpr std::array<Eigen::Index, 3> vy_idx_{1, 4, 7};
  constexpr std::array<Eigen::Index, 3> p_idx_{2, 5, 8};
  // Place the element matrix for $-\Delta$ in the element matrix
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      AK(vx_idx_[i], vx_idx_[j]) = L(i, j);
      AK(vy_idx_[i], vy_idx_[j]) = L(i, j);
    }
  }
  // Insert contributions to "pressure parts" of element matrix
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      AK(vx_idx_[j], p_idx_[i]) = AK(p_idx_[i], vx_idx_[j]) = G(0, j) / 3.0;
      AK(vy_idx_[j], p_idx_[i]) = AK(p_idx_[i], vy_idx_[j]) = G(1, j) / 3.0;
    }
  }
  // Finally scale with the area of the triangle
  AK *= area;
  return AK;
}
/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_3 */
MINIElementMatrixProvider::ElemMat MINIElementMatrixProvider::Eval(
    const lf::mesh::Entity& cell) {
  LF_VERIFY_MSG(cell.RefEl() == lf::base::RefEl::kTria(),
                "Unsupported cell type " << cell.RefEl());
  // Area of the triangle
  double area = lf::geometry::Volume(*cell.Geometry());
  // Compute gradients of barycentric coordinate functions, see
  // \lref{cpp:gradbarycoords}. First fetch vertices of the triangle
  auto endpoints = lf::geometry::Corners(*(cell.Geometry()));
  Eigen::Matrix<double, 3, 3> X;  // temporary matrix
  X.block<3, 1>(0, 0) = Eigen::Vector3d::Ones();
  X.block<3, 2>(0, 1) = endpoints.transpose();
  // This matrix contains $\cob{\grad \lambda_i}$ in its columns
  const auto G{X.inverse().block<2, 3>(1, 0)};
  // Inner products of gradients of barycentric coordinate functions
  // (Scaled) element matrix for $-\Delta$
  const Eigen::Matrix3d L{G.transpose() * G};
  // Fixed-size matrix containing the element matrix
  ElemMat AK;
  AK.setZero();
  // indices of local shape  function for various solution components
  constexpr std::array<Eigen::Index, 3> vx_idx_{0, 3, 6};
  constexpr std::array<Eigen::Index, 3> vy_idx_{1, 4, 7};
  constexpr std::array<Eigen::Index, 3> p_idx_{2, 5, 8};
  // Place the element matrix for $-\Delta$ in the element matrix
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      AK(vx_idx_[i], vx_idx_[j]) = L(i, j);
      AK(vy_idx_[i], vy_idx_[j]) = L(i, j);
    }
  }
  // Insert contributions to "pressure parts" of element matrix
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      AK(vx_idx_[j], p_idx_[i]) = AK(p_idx_[i], vx_idx_[j]) = G(0, j) / 3.0;
      AK(vy_idx_[j], p_idx_[i]) = AK(p_idx_[i], vy_idx_[j]) = G(1, j) / 3.0;
    }
  }
  // Finally scale with the area of the triangle
  AK *= area;
  return AK;
}
/* SAM_LISTING_END_3 */

/* SAM_LISTING_BEGIN_7 */
void testCvgSimpleFEM(unsigned int refsteps) {
  // ********** Part I: Manufactured solution  **********
  // Analytic solution for velocity and pressure
  Eigen::Vector2d d{2.0, 1.0};
  auto v_ex = [&d](Eigen::Vector2d x) -> Eigen::Vector2d {
    x += Eigen::Vector2d(0.5, 0.5);
    const double nx = x.norm();
    return 0.25 * (-std::log(nx) * d + x * (x.dot(d)) / (nx * nx));
  };
  auto p_ex = [&d](Eigen::Vector2d x) -> double {
    x += Eigen::Vector2d(0.5, 0.5);
    return -0.5 * x.dot(d) / x.squaredNorm() + (d[0] + d[1]) * 0.5 * 0.502128;
  };
  auto grad_v1 = [&d](Eigen::Vector2d x) -> Eigen::Vector2d {
    x += Eigen::Vector2d(0.5, 0.5);
    const double x2 = x[0] * x[0];
    const double x3 = x2 * x[0];
    const double y2 = x[1] * x[1];
    const double den = (x2 + y2) * (x2 + y2);
    return 0.25 * d[0] *
               Eigen::Vector2d{(-x3 + x[0] * y2) / den,
                               -x[1] * (3 * x2 + y2) / den} +
           0.25 * d[1] *
               Eigen::Vector2d{x[1] * (-x2 + y2) / den, x[0] * (x2 - y2) / den};
  };
  auto grad_v2 = [&d](Eigen::Vector2d x) -> Eigen::Vector2d {
    x += Eigen::Vector2d(0.5, 0.5);
    const double x2 = x[0] * x[0];
    const double x3 = x2 * x[0];
    const double y2 = x[1] * x[1];
    const double den = (x2 + y2) * (x2 + y2);
    return 0.25 * d[1] *
               Eigen::Vector2d{-x[0] * (x2 + 3 * y2) / den,
                               x[1] * (x2 - y2) / den} +
           0.25 * d[0] *
               Eigen::Vector2d{x[1] * (-x2 + y2) / den, x[0] * (x2 - y2) / den};
  };
  // ********** Part II: Loop over sequence of meshes **********
  // Generate a small unstructured triangular mesh
  const std::shared_ptr<lf::mesh::Mesh> mesh_ptr =
      lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1.0 / 3.0);
  const std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh_ptr,
                                                              refsteps);
  lf::refinement::MeshHierarchy& multi_mesh{*multi_mesh_p};
  // Ouput summary information about hierarchy of nested meshes
  std::cout << "\t Sequence of nested meshes created\n";
  multi_mesh.PrintInfo(std::cout);

  // Number of levels
  const int L = multi_mesh.NumLevels();

  // Table of various error norms
  std::vector<std::tuple<size_t, double, double, double, double, double>> errs;
  for (int level = 0; level < L; ++level) {
    const std::shared_ptr<const lf::mesh::Mesh> lev_mesh_p =
        multi_mesh.getMesh(level);
    // Define Lagranggian FE spaces for piecewise linear approximation
    auto fes_o1_ptr =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(lev_mesh_p);
    // Fetch dof handler for the components of the velocity
    const lf::assemble::DofHandler& dofh_u = fes_o1_ptr->LocGlobMap();
    //  Fetch dof handler for the pressure (the same as for velocity)
    const lf::assemble::DofHandler& dofh_p = fes_o1_ptr->LocGlobMap();
    // ********** Part III: Solving on a single mesh **********
    // Initialize dof handler for simple FEM
    lf::assemble::UniformFEDofHandler dofh(lev_mesh_p,
                                           {{lf::base::RefEl::kPoint(), 3},
                                            {lf::base::RefEl::kSegment(), 0},
                                            {lf::base::RefEl::kTria(), 0},
                                            {lf::base::RefEl::kQuad(), 0}});
    LF_ASSERT_MSG(dofh.NumDofs() == 2 * dofh_u.NumDofs() + dofh_p.NumDofs(),
                  "No dof mismatch");
    // Build and solve the linear system with trace of the exact velocity
    // solution as Dirichlet data.
    // Total number of d.o.f. in monolithic FE spaces
    size_t n = dofh.NumDofs();
    std::cout << "Computing with " << n << " d.o.f.s, assembling LSE .. "
              << std::flush;
    // Obtain full Galerkin matrix in triplet format
    // Full Galerkin matrix in triplet format taking into account the zero mean
    // constraint on the pressure.
    lf::assemble::COOMatrix<double> A(n + 1, n + 1);
    // Set up computation of element matrix
    SimpleFEMElementMatrixProvider themp{};
    // Assemble \cor{full} Galerkin matrix for simple Stokes  FEM
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, themp, A);

    // Add bottom row and right column corresponding to Lagrange multiplier
    for (const lf::mesh::Entity* cell : dofh.Mesh()->Entities(0)) {
      LF_ASSERT_MSG(cell->RefEl() == lf::base::RefEl::kTria(),
                    "Only implemented for triangles");
      // Obtain area of triangle
      const double area = lf::geometry::Volume(*cell->Geometry());
      // The pressure GSFs are associated with the nodes
      const std::span<const lf::mesh::Entity* const> nodes{
          cell->SubEntities(2)};
      // Loop over nodes
      for (const lf::mesh::Entity* node : nodes) {
        // Area of the cell
        // Obtain index of tent function associated with node
        // All indices of global shape functions sitting at node
        std::span<const lf::assemble::gdof_idx_t> dof_idx{
            dofh.InteriorGlobalDofIndices(*node)};
        LF_ASSERT_MSG(dof_idx.size() == 3, "Node must carry 3 dofs!");
        // The index of the tent function is the third one
        const lf::assemble::gdof_idx_t tent_idx = dof_idx[2];
        A.AddToEntry(n, tent_idx, area / 3.0);
        A.AddToEntry(tent_idx, n, area / 3.0);
      }
    }
    // Auxiliary right-hand side vector
    Eigen::VectorXd phi(A.cols());
    phi.setZero();
    // Impose Dirichlet boundary conditions
    const std::shared_ptr<const lf::mesh::Mesh> mesh_p = dofh.Mesh();
    // Flag nodes located on the boundary
    auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 2)};
    // Flag vector for d.o.f. on the boundary
    std::vector<std::pair<bool, double>> ess_dof_select(n + 1, {false, 0.0});
    // Visit nodes on the boundary
    for (const lf::mesh::Entity* node : mesh_p->Entities(2)) {
      if (bd_flags(*node)) {
        // Indices of global shape functions sitting at node
        std::span<const lf::assemble::gdof_idx_t> dof_idx{
            dofh.InteriorGlobalDofIndices(*node)};
        LF_ASSERT_MSG(dof_idx.size() == 3, "Node must carry 3 dofs!");
        // Position of node
        const Eigen::Vector2d pos{Corners(*(node->Geometry())).col(0)};
        // Dirichlet data
        const Eigen::Vector2d g_val{v_ex(pos)};
        // x-component of the velocity
        ess_dof_select[dof_idx[0]] = {true, g_val[0]};
        // y-component of the velocity
        ess_dof_select[dof_idx[1]] = {true, g_val[1]};
      }
    }
    // modify linear system of equations
    lf::assemble::FixFlaggedSolutionComponents<double>(
        [&ess_dof_select](lf::assemble::glb_idx_t dof_idx)
            -> std::pair<bool, double> { return ess_dof_select[dof_idx]; },
        A, phi);
    // Assembly completed: Convert COO matrix A into CRS format using Eigen's
    // internal conversion routines.
    Eigen::SparseMatrix<double> A_crs{A.makeSparse()};

    std::cout << "done. Solving ..... " << std::flush;
    // Solve linear system using Eigen's sparse direct elimination
    // Examine return status of solver in case the matrix is singular
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A_crs);
    LF_VERIFY_MSG(solver.info() == Eigen::Success, "LU decomposition failed");
    const Eigen::VectorXd res = solver.solve(phi);
    LF_VERIFY_MSG(solver.info() == Eigen::Success, "Solving LSE failed");
    std::cout << "done. |dof vector| = " << res.norm() << std::endl;
    // This is the coefficient vector for the FE solution; Dirichlet
    // boundary conditions are included

    // Coefficient vectors for the first and second component of the velocity
    Eigen::VectorXd coeff_vec_u1 = Eigen::VectorXd::Zero(dofh_u.NumDofs());
    Eigen::VectorXd coeff_vec_u2 = Eigen::VectorXd::Zero(dofh_u.NumDofs());
    // Coefficient vector for the pressure
    Eigen::VectorXd coeff_vec_p = Eigen::VectorXd::Zero(dofh_p.NumDofs());

    // ********** Part IV: Compute error norms **********
    // Remapping dofs in orde to be able to use MeshFunctionFE
    for (auto e : lev_mesh_p->Entities(2)) {
      // Global indices for u1, u2 for the respective vertex or edge
      auto glob_idxs = dofh.InteriorGlobalDofIndices(*e);
      auto glob_idx_o1 = dofh_u.InteriorGlobalDofIndices(*e)[0];
      // Extract the correct elements for the coefficient vector of the
      // components of u and the pressure p
      coeff_vec_u1[glob_idx_o1] = res[glob_idxs[0]];
      coeff_vec_u2[glob_idx_o1] = res[glob_idxs[1]];
      coeff_vec_p(glob_idx_o1) = res[glob_idxs[2]];
    }
    // Variables for storing the error norms
    double L2err_u1, L2err_u2, H1err_u1, H1err_u2, L2err_p;

    // Define finite-element mesh functions
    const lf::fe::MeshFunctionFE mf_o2_u1(fes_o1_ptr, coeff_vec_u1);
    const lf::fe::MeshFunctionFE mf_o2_u2(fes_o1_ptr, coeff_vec_u2);
    const lf::fe::MeshFunctionFE mf_o1_p(fes_o1_ptr, coeff_vec_p);
    const lf::fe::MeshFunctionGradFE mf_o2_grad_u1(fes_o1_ptr, coeff_vec_u1);
    const lf::fe::MeshFunctionGradFE mf_o2_grad_u2(fes_o1_ptr, coeff_vec_u2);

    // Exact solution for the first component of the velocity
    auto u1 = [&v_ex](Eigen::Vector2d x) -> double { return v_ex(x)[0]; };
    const lf::mesh::utils::MeshFunctionGlobal mf_u1{u1};
    // Exact solution for the gradient of $v_1$
    const lf::mesh::utils::MeshFunctionGlobal mf_grad_u1{grad_v1};
    // Exact solution second component of  the velocity
    auto u2 = [&v_ex](Eigen::Vector2d x) -> double { return v_ex(x)[1]; };
    const lf::mesh::utils::MeshFunctionGlobal mf_u2{u2};
    // Exact solution for the gradient of $v_2$
    const lf::mesh::utils::MeshFunctionGlobal mf_grad_u2{grad_v2};
    // Mesh function for exact solution pressure
    const lf::mesh::utils::MeshFunctionGlobal mf_p{p_ex};
    // compute errors with 5th order quadrature rules
    L2err_u1 = std::sqrt(lf::fe::IntegrateMeshFunction(
        *lev_mesh_p, lf::mesh::utils::squaredNorm(mf_o2_u1 - mf_u1), 4));
    L2err_u2 = std::sqrt(lf::fe::IntegrateMeshFunction(
        *lev_mesh_p, lf::mesh::utils::squaredNorm(mf_o2_u2 - mf_u2), 4));
    H1err_u1 = std::sqrt(lf::fe::IntegrateMeshFunction(
        *lev_mesh_p, lf::mesh::utils::squaredNorm(mf_o2_grad_u1 - mf_grad_u1),
        4));
    H1err_u2 = std::sqrt(lf::fe::IntegrateMeshFunction(
        *lev_mesh_p, lf::mesh::utils::squaredNorm(mf_o2_grad_u2 - mf_grad_u2),
        4));
    L2err_p = std::sqrt(lf::fe::IntegrateMeshFunction(
        *lev_mesh_p, lf::mesh::utils::squaredNorm(mf_o1_p - mf_p), 4));
    errs.emplace_back(dofh.NumDofs(), L2err_u1, L2err_u2, H1err_u1, H1err_u2,
                      L2err_p);
  }
  // Output table of errors to file and terminal
  std::ofstream out_file("errors.txt");
  std::cout.precision(3);
  std::cout << std::endl
            << std::left << std::setw(10) << "N" << std::right << std::setw(16)
            << "L2 err(v1)" << std::setw(16) << "L2 err(v2)" << std::setw(16)
            << "H1 err(v1)" << std::setw(16) << "H1 err(v2)" << std::setw(16)
            << "L2 err(p)" << '\n';
  std::cout << "---------------------------------------------" << '\n';
  for (const auto& err : errs) {
    auto [N, L2err_u1, L2err_u2, H1err_u1, H1err_u2, L2err_p] = err;
    out_file << std::left << std::setw(10) << N << std::left << std::setw(16)
             << L2err_u1 << std::setw(16) << L2err_u2 << std::setw(16)
             << H1err_u1 << std::setw(16) << H1err_u2 << std::setw(16)
             << L2err_p << '\n';
    std::cout << std::left << std::setw(10) << N << std::left << std::setw(16)
              << L2err_u1 << std::setw(16) << L2err_u2 << std::setw(16)
              << H1err_u1 << std::setw(16) << H1err_u2 << std::setw(16)
              << L2err_p << '\n';
  }
}
/* SAM_LISTING_END_7 */

void testCvgMINIFEM(unsigned int refsteps) {
  // ********** Part I: Manufactured solution  **********
  // Analytic solution for velocity and pressure
  Eigen::Vector2d d{2.0, 1.0};
  auto v_ex = [&d](Eigen::Vector2d x) -> Eigen::Vector2d {
    x += Eigen::Vector2d(0.5, 0.5);
    const double nx = x.norm();
    return 0.25 * (-std::log(nx) * d + x * (x.dot(d)) / (nx * nx));
  };
  auto p_ex = [&d](Eigen::Vector2d x) -> double {
    x += Eigen::Vector2d(0.5, 0.5);
    return -0.5 * x.dot(d) / x.squaredNorm() + (d[0] + d[1]) * 0.5 * 0.502128;
  };
  auto grad_v1 = [&d](Eigen::Vector2d x) -> Eigen::Vector2d {
    x += Eigen::Vector2d(0.5, 0.5);
    const double x2 = x[0] * x[0];
    const double x3 = x2 * x[0];
    const double y2 = x[1] * x[1];
    const double den = (x2 + y2) * (x2 + y2);
    return 0.25 * d[0] *
               Eigen::Vector2d{(-x3 + x[0] * y2) / den,
                               -x[1] * (3 * x2 + y2) / den} +
           0.25 * d[1] *
               Eigen::Vector2d{x[1] * (-x2 + y2) / den, x[0] * (x2 - y2) / den};
  };
  auto grad_v2 = [&d](Eigen::Vector2d x) -> Eigen::Vector2d {
    x += Eigen::Vector2d(0.5, 0.5);
    const double x2 = x[0] * x[0];
    const double x3 = x2 * x[0];
    const double y2 = x[1] * x[1];
    const double den = (x2 + y2) * (x2 + y2);
    return 0.25 * d[1] *
               Eigen::Vector2d{-x[0] * (x2 + 3 * y2) / den,
                               x[1] * (x2 - y2) / den} +
           0.25 * d[0] *
               Eigen::Vector2d{x[1] * (-x2 + y2) / den, x[0] * (x2 - y2) / den};
  };
  // ********** Part II: Loop over sequence of meshes **********
  // Generate a small unstructured triangular mesh
  const std::shared_ptr<lf::mesh::Mesh> mesh_ptr =
      lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1.0 / 3.0);
  const std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh_ptr,
                                                              refsteps);
  lf::refinement::MeshHierarchy& multi_mesh{*multi_mesh_p};
  // Ouput summary information about hierarchy of nested meshes
  std::cout << "\t Sequence of nested meshes created\n";
  multi_mesh.PrintInfo(std::cout);

  // Number of levels
  const int L = multi_mesh.NumLevels();

  // Table of various error norms
  std::vector<std::tuple<size_t, double, double, double, double, double>> errs;
  for (int level = 0; level < L; ++level) {
    const std::shared_ptr<const lf::mesh::Mesh> lev_mesh_p =
        multi_mesh.getMesh(level);
    // Define Lagranggian FE spaces for piecewise linear approximation
    auto fes_o1_ptr =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(lev_mesh_p);
    // Fetch dof handler for the components of the velocity
    const lf::assemble::DofHandler& dofh_u = fes_o1_ptr->LocGlobMap();
    //  Fetch dof handler for the pressure (the same as for velocity)
    const lf::assemble::DofHandler& dofh_p = fes_o1_ptr->LocGlobMap();
    // ********** Part III: Solving on a single mesh **********
    // Initialize dof handler for simple FEM
    lf::assemble::UniformFEDofHandler dofh(lev_mesh_p,
                                           {{lf::base::RefEl::kPoint(), 3},
                                            {lf::base::RefEl::kSegment(), 0},
                                            {lf::base::RefEl::kTria(), 0},
                                            {lf::base::RefEl::kQuad(), 0}});
    // Build and solve the linear system with trace of the exact velocity
    // solution as Dirichlet data.
    // Total number of d.o.f. in monolithic FE spaces
    size_t n = dofh.NumDofs();
    std::cout << "Computing with " << n << " d.o.f.s, assembling LSE .. "
              << std::flush;
    // Obtain full Galerkin matrix in triplet format
    // Full Galerkin matrix in triplet format taking into account the zero mean
    // constraint on the pressure.
    lf::assemble::COOMatrix<double> A(n + 1, n + 1);
    // Set up computation of element matrix
    SimpleFEMElementMatrixProvider themp{};
    // Assemble \cor{full} Galerkin matrix for simple Stokes  FEM
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, themp, A);

    // Add bottom row and right column corresponding to Lagrange multiplier
    for (const lf::mesh::Entity* cell : dofh.Mesh()->Entities(0)) {
      LF_ASSERT_MSG(cell->RefEl() == lf::base::RefEl::kTria(),
                    "Only implemented for triangles");
      // Obtain area of triangle
      const double area = lf::geometry::Volume(*cell->Geometry());
      // The pressure GSFs are associated with the nodes
      const std::span<const lf::mesh::Entity* const> nodes{
          cell->SubEntities(2)};
      // Loop over nodes
      for (const lf::mesh::Entity* node : nodes) {
        // Area of the cell
        // Obtain index of tent function associated with node
        // All indices of global shape functions sitting at node
        std::span<const lf::assemble::gdof_idx_t> dof_idx{
            dofh.InteriorGlobalDofIndices(*node)};
        LF_ASSERT_MSG(dof_idx.size() == 3, "Node must carry 3 dofs!");
        // The index of the tent function is the third one
        const lf::assemble::gdof_idx_t tent_idx = dof_idx[2];
        A.AddToEntry(n, tent_idx, area / 3.0);
        A.AddToEntry(tent_idx, n, area / 3.0);
      }
    }
    // Auxiliary right-hnad side vector
    Eigen::VectorXd phi(A.cols());
    phi.setZero();
    // Impose Dirichlet boundary conditions
    const std::shared_ptr<const lf::mesh::Mesh> mesh_p = dofh.Mesh();
    // Flag nodes located on the boundary
    auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 2)};
    // Flag vector for d.o.f. on the boundary
    std::vector<std::pair<bool, double>> ess_dof_select(n + 1, {false, 0.0});
    // Visit nodes on the boundary
    for (const lf::mesh::Entity* node : mesh_p->Entities(2)) {
      if (bd_flags(*node)) {
        // Indices of global shape functions sitting at node
        std::span<const lf::assemble::gdof_idx_t> dof_idx{
            dofh.InteriorGlobalDofIndices(*node)};
        // Position of node
        const Eigen::Vector2d pos{Corners(*(node->Geometry())).col(0)};
        // Dirichlet data
        const Eigen::Vector2d g_val{v_ex(pos)};
        // x-component of the velocity
        ess_dof_select[dof_idx[0]] = {true, g_val[0]};
        // y-component of the velocity
        ess_dof_select[dof_idx[1]] = {true, g_val[1]};
      }
    }
    // modify linear system of equations
    lf::assemble::FixFlaggedSolutionComponents<double>(
        [&ess_dof_select](lf::assemble::glb_idx_t dof_idx)
            -> std::pair<bool, double> { return ess_dof_select[dof_idx]; },
        A, phi);
    // Assembly completed: Convert COO matrix A into CRS format using Eigen's
    // internal conversion routines.
    Eigen::SparseMatrix<double> A_crs{A.makeSparse()};

    std::cout << "done. Solving ..... " << std::flush;
    // Solve linear system using Eigen's sparse direct elimination
    // Examine return status of solver in case the matrix is singular
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A_crs);
    LF_VERIFY_MSG(solver.info() == Eigen::Success, "LU decomposition failed");
    // This is the coefficient vector for the FE solution; Dirichlet
    // boundary conditions are included
    const Eigen::VectorXd res = solver.solve(phi);
    LF_VERIFY_MSG(solver.info() == Eigen::Success, "Solving LSE failed");
    std::cout << "done. |dof vector| = " << res.norm() << std::endl;
/* SAM_LISTING_END_8 */
    // ********** Part IV: Compute error norms **********
    // Coefficient vectors for the first and second component of the velocity
    Eigen::VectorXd coeff_vec_u1 = Eigen::VectorXd::Zero(dofh_u.NumDofs());
    Eigen::VectorXd coeff_vec_u2 = Eigen::VectorXd::Zero(dofh_u.NumDofs());
    // Coefficient vector for the pressure
    Eigen::VectorXd coeff_vec_p = Eigen::VectorXd::Zero(dofh_p.NumDofs());
    // Remapping dofs in orde to be able to use MeshFunctionFE
    for (auto e : lev_mesh_p->Entities(2)) {
      // Global indices for u1, u2 for the respective vertex or edge
      auto glob_idxs = dofh.InteriorGlobalDofIndices(*e);
      auto glob_idx_o1 = dofh_u.InteriorGlobalDofIndices(*e)[0];
      // Extract the correct elements for the coefficient vector of the
      // components of u and the pressure p
      coeff_vec_u1[glob_idx_o1] = res[glob_idxs[0]];
      coeff_vec_u2[glob_idx_o1] = res[glob_idxs[1]];
      coeff_vec_p(glob_idx_o1) = res[glob_idxs[2]];
    }
    // Variables for storing the error norms
    double L2err_u1, L2err_u2, H1err_u1, H1err_u2, L2err_p;

    // Define finite-element mesh functions
    const lf::fe::MeshFunctionFE mf_o2_u1(fes_o1_ptr, coeff_vec_u1);
    const lf::fe::MeshFunctionFE mf_o2_u2(fes_o1_ptr, coeff_vec_u2);
    const lf::fe::MeshFunctionFE mf_o1_p(fes_o1_ptr, coeff_vec_p);
    const lf::fe::MeshFunctionGradFE mf_o2_grad_u1(fes_o1_ptr, coeff_vec_u1);
    const lf::fe::MeshFunctionGradFE mf_o2_grad_u2(fes_o1_ptr, coeff_vec_u2);

    // Exact solution for the first component of the velocity
    auto u1 = [&v_ex](Eigen::Vector2d x) -> double { return v_ex(x)[0]; };
    const lf::mesh::utils::MeshFunctionGlobal mf_u1{u1};
    // Exact solution for the gradient of $v_1$
    const lf::mesh::utils::MeshFunctionGlobal mf_grad_u1{grad_v1};
    // Exact solution second component of  the velocity
    auto u2 = [&v_ex](Eigen::Vector2d x) -> double { return v_ex(x)[1]; };
    const lf::mesh::utils::MeshFunctionGlobal mf_u2{u2};
    // Exact solution for the gradient of $v_2$
    const lf::mesh::utils::MeshFunctionGlobal mf_grad_u2{grad_v2};
    // Mesh function for exact solution pressure
    const lf::mesh::utils::MeshFunctionGlobal mf_p{p_ex};
    // compute errors with 5th order quadrature rules
    L2err_u1 = std::sqrt(lf::fe::IntegrateMeshFunction(
        *lev_mesh_p, lf::mesh::utils::squaredNorm(mf_o2_u1 - mf_u1), 4));
    L2err_u2 = std::sqrt(lf::fe::IntegrateMeshFunction(
        *lev_mesh_p, lf::mesh::utils::squaredNorm(mf_o2_u2 - mf_u2), 4));
    H1err_u1 = std::sqrt(lf::fe::IntegrateMeshFunction(
        *lev_mesh_p, lf::mesh::utils::squaredNorm(mf_o2_grad_u1 - mf_grad_u1),
        4));
    H1err_u2 = std::sqrt(lf::fe::IntegrateMeshFunction(
        *lev_mesh_p, lf::mesh::utils::squaredNorm(mf_o2_grad_u2 - mf_grad_u2),
        4));
    L2err_p = std::sqrt(lf::fe::IntegrateMeshFunction(
        *lev_mesh_p, lf::mesh::utils::squaredNorm(mf_o1_p - mf_p), 4));
    errs.emplace_back(dofh.NumDofs(), L2err_u1, L2err_u2, H1err_u1, H1err_u2,
                      L2err_p);
  }
  // Output table of errors to file and terminal
  std::ofstream out_file("errors.txt");
  std::cout.precision(3);
  std::cout << std::endl
            << std::left << std::setw(10) << "N" << std::right << std::setw(16)
            << "L2 err(v1)" << std::setw(16) << "L2 err(v2)" << std::setw(16)
            << "H1 err(v1)" << std::setw(16) << "H1 err(v2)" << std::setw(16)
            << "L2 err(p)" << '\n';
  std::cout << "---------------------------------------------" << '\n';
  for (const auto& err : errs) {
    auto [N, L2err_u1, L2err_u2, H1err_u1, H1err_u2, L2err_p] = err;
    out_file << std::left << std::setw(10) << N << std::left << std::setw(16)
             << L2err_u1 << std::setw(16) << L2err_u2 << std::setw(16)
             << H1err_u1 << std::setw(16) << H1err_u2 << std::setw(16)
             << L2err_p << '\n';
    std::cout << std::left << std::setw(10) << N << std::left << std::setw(16)
              << L2err_u1 << std::setw(16) << L2err_u2 << std::setw(16)
              << H1err_u1 << std::setw(16) << H1err_u2 << std::setw(16)
              << L2err_p << '\n';
  }
}

}  // namespace StokesMINIElement
