// Gmsh description of pipe geometry
// Target mesh size
h = 0.006025;
// Corners
Point(1) = { 0, 1, 0, h };
Point(2) = { 0, 0.5, 0, h };
Point(3) = { 0.33, 0.5, 0, h };
Point(4) = { 0.33, 0, 0, h };
Point(5) = { 1, 0, 0, h };
Point(6) = { 1, 0.5, 0, h };
Point(7) = { 0.66, 0.5, 0, h };
Point(8) = { 0.66, 1, 0, h };
// Edges
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};
// Domain
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};	
Plane Surface(1) = {1};
