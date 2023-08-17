/**
 * @file semilinearellipticbvp_test.cc
 * @brief NPDE homework semilinearellipticbvp code
 * @author R. Hiptmair
 * @date June 2023
 * @copyright Developed at SAM, ETH Zurich
 */

#include "../semilinearellipticbvp.h"


#include <lf/mesh/test_utils/test_meshes.h>

#include <gtest/gtest.h>

namespace semilinearellipticbvp::test {
    TEST(semilinearellipticbvp, newtonSolveSemilinearBVP)
    {
        // We perform a convergence analysis using a manufactured solution

        // The number of refinements of the mesh.
        int reflevels = 5;

        // Manufactured solution of $-\Delta u + \sinh(u) = f$ with homogeneous
        // Dirichlet boundary conditions
        auto u = [](Eigen::Vector2d x) -> double {
            return std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]);
        };
        auto grad_u = [](Eigen::Vector2d x) -> Eigen::Vector2d {
            return M_PI *
                Eigen::Vector2d(std::cos(M_PI * x[0]) * std::sin(M_PI * x[1]),
                                std::sin(M_PI * x[0]) * std::cos(M_PI * x[1]));
        };
        auto f = [&u](Eigen::Vector2d x) -> double {
            return 2.0 * M_PI * M_PI * u(x) + std::sinh(u(x));
        };
        // Wrap lambdas into mesh functions
        lf::mesh::utils::MeshFunctionGlobal mf_u{u};
        lf::mesh::utils::MeshFunctionGlobal mf_grad_u{grad_u};
        lf::mesh::utils::MeshFunctionGlobal mf_f{f};

        // Triangular mesh hierarachy of unit square for testing
        // Adapted from Lehrfem++ \cppfile{homDir_linfe_demo.cc}
        std::shared_ptr<lf::mesh::Mesh> mesh_p =
            lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1.0 / 3.0);
        std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p =
            lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh_p,
                                                                    reflevels);
        lf::refinement::MeshHierarchy& multi_mesh{*multi_mesh_p};
        std::size_t L = multi_mesh.NumLevels();  // Number of levels

        // Vector for keeping error norms
        std::vector<std::tuple<int, double, double>> errs{};
        // LEVEL LOOP: Do computations on all levels
        for (int level = 0; level < L; ++level) {
            mesh_p = multi_mesh.getMesh(level);
            // Set up global FE space; lowest order Lagrangian finite elements
            auto fe_space =
                std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);
            // Compute finite-element solution of boundary value problem
            const Eigen::VectorXd sol_vec = newtonSolveSemilinearBVP(fe_space, f);

            // Compute error norms
            const lf::fe::MeshFunctionFE mf_sol(fe_space, sol_vec);
            const lf::fe::MeshFunctionGradFE mf_grad_sol(fe_space, sol_vec);
            // compute errors with 3rd order quadrature rules, which is sufficient for
            // piecewise linear finite elements
            double L2err =  // NOLINT
                std::sqrt(lf::fe::IntegrateMeshFunction(
                    *mesh_p, lf::mesh::utils::squaredNorm(mf_sol - mf_u), 2));
            double H1serr = std::sqrt(lf::fe::IntegrateMeshFunction(  // NOLINT
                *mesh_p, lf::mesh::utils::squaredNorm(mf_grad_sol - mf_grad_u), 2));
            errs.emplace_back(mesh_p->NumEntities(2), L2err, H1serr);
        }
        // Output table of errors
        for (int i=3; i<reflevels+1; ++i) {
            auto [N, l2err, h1serr] = errs.at(i);
            auto [Nprev, l2errprev, h1serrprev] = errs.at(i-1);

            // We expect second-order convergence in L2
            EXPECT_GE(std::log(l2errprev/l2err)/std::log(2.),1.95);

            // We expect first-order convergence in H1
            EXPECT_GE(std::log(h1serrprev/h1serr)/std::log(2.),0.95);
        }
    }


}  // namespace semilinearellipticbvp::test
