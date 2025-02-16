# This file specifies additional data for the dependencies that are imported via hunter

hunter_config(lehrfempp
  URL "https://github.com/craffael/lehrfempp/archive/741ee74b3456d7764c74731e1cfa2fa6588ce9ce.tar.gz"
  SHA1 "9baeb103f2d40ef6d41cdf766a6504665b89489b"
  CMAKE_ARGS LF_REDIRECT_ASSERTS=Off
)

hunter_config(Eigen VERSION 3.4.0)
hunter_config(Boost VERSION 1.78.0)
