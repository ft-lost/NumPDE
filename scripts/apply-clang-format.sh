#! /bin/bash

# Very basic script, must be run from the root of the repository (and only on Linux)
find developers lecturecodes include -iname *.h -o -iname *.hpp -o -iname *.c -o -iname *.cpp -o -iname *.cc | xargs clang-format -i