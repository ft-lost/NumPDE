
#ifndef MESH_TRIANGLE_UNIT_SQUARE_EIGEN
#define MESH_TRIANGLE_UNIT_SQUARE_EIGEN

#include <Eigen/Dense>

Eigen::MatrixXd generateMesh(int M)
{
    double h = 1.0/(M+1);
    
    Eigen::MatrixXd nodes((M+2)*(M+2), 2);
    for(int i=0; i<M+2; ++i) {
        for(int j=0; j<M+2; ++j) {
            nodes(i*(M+2)+j, 0) = i*h;
            nodes(i*(M+2)+j, 1) = j*h;
        }
    }

    Eigen::MatrixXd elements(2*(M+1)*(M+1), 6);
    int cnt = 0;
    for(int i=0; i<M+1; ++i) {
        for(int j=0; j<M+1; ++j) {
            elements.row(cnt) << nodes.row(i*(M+2) + j), nodes.row((i+1)*(M+2) + j),   nodes.row((i+1)*(M+2) + j+1);
            ++cnt;
            elements.row(cnt) << nodes.row(i*(M+2) + j), nodes.row((i+1)*(M+2) + j+1), nodes.row(i*(M+2) + j+1);
            ++cnt;
        }
    }

    return elements;
}

#endif // MESH_TRIANGLE_UNIT_SQUARE_EIGEN
