# This file specifies additional data for the dependencies that are imported via hunter

hunter_config(lehrfempp
  URL "https://github.com/craffael/lehrfempp/archive/release-0.9.0.tar.gz"
  SHA1 "825ac9dd9de8e7e86564ce01da4d1423cc54ccbf"
  CMAKE_ARGS LF_REDIRECT_ASSERTS=Off
)
hunter_config(Eigen VERSION 3.4.0)
