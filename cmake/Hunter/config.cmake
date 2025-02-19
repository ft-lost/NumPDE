# This file specifies additional data for the dependencies that are imported via hunter

hunter_config(lehrfempp
  URL "https://github.com/craffael/lehrfempp/archive/release-0.9.2.tar.gz"
  SHA1 "a37ec9deb7427d8cbec0f6c7ca48320398a7d417"
  CMAKE_ARGS LF_REDIRECT_ASSERTS=Off
)

hunter_config(Eigen VERSION 3.4.0)
hunter_config(Boost VERSION 1.78.0)

hunter_config(Boost 
  VERSION 1.86.0
  URL "https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.bz2"
  SHA1 "fd0d26a7d5eadf454896942124544120e3b7a38f"
)
