/**
 * @ file boundarylength.cc
 * @ brief NPDE homework LengthOfBoundary code
 * @ author Christian Mitsch
 * @ date 03.03.2019
 * @ copyright Developed at ETH Zurich
 */

#include "boundarylength.h"

#include <lf/geometry/geometry.h>
#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/utils/utils.h>

namespace LengthOfBoundary {

/* SAM_LISTING_BEGIN_1 */
double volumeOfDomain(const std::shared_ptr<lf::mesh::Mesh> mesh_p) {
  double volume = 0.0;
  //====================
  // Your code goes here
  //====================
  const lf::mesh::Mesh &mesh{*mesh_p};

  for(const lf::mesh::Entity *ent : mesh.Entities(0)){
    const lf::geometry::Geometry *geo = ent->Geometry();
    volume += lf::geometry::Volume(*geo);
  }

  return volume;
}
/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_2 */
double lengthOfBoundary(const std::shared_ptr<lf::mesh::Mesh> mesh_p) {
  double length = 0.0;
  //====================
  // Your code goes here
  //====================
  lf::mesh::utils::CodimMeshDataSet<bool> bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p,1)};
  const lf::mesh::Mesh &mesh{*mesh_p};

  for(const lf::mesh::Entity *ent : mesh.Entities(1)){
    const lf::geometry::Geometry *geo = ent->Geometry();

    if(bd_flags(*ent)) length += lf::geometry::Volume(*geo);

  }

  return length;
}
/* SAM_LISTING_END_2 */

/* SAM_LISTING_BEGIN_3 */
std::pair<double, double> measureDomain(std::string filename) {
  double volume, length;

  //====================
  // Your code goes here
  //====================
  auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(mesh_factory), filename);
  const std::shared_ptr<lf::mesh::Mesh> mesh_p = reader.mesh();
  volume = volumeOfDomain(mesh_p);
  length = lengthOfBoundary(mesh_p);

  return {volume, length};
}
/* SAM_LISTING_END_3 */

}  // namespace LengthOfBoundary
