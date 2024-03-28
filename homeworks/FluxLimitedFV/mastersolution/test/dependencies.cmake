# Dependencies of mastersolution:

# DIR will be provided by the calling file.

set(SOURCES
  ${DIR}/test/fluxlimitedfv_test.cc
)

# Libraries to be used. If the code does not rely on LehrFEM++
# all the libraries LF:* can be removed 
set(LIBRARIES
  Eigen3::Eigen
  GTest::gtest_main
)

