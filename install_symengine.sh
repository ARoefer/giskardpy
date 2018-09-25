#!/usr/bin/env bash
sudo apt-get install llvm-6.0-dev
git clone https://github.com/symengine/symengine.git
git clone https://github.com/symengine/symengine.py.git
cd symengine
git checkout `cat ../symengine.py/symengine_version.txt`
cmake -DBUILD_SHARED_LIBS:BOOL=ON -DWITH_LLVM:BOOL=ON .
make
sudo make install
cd ../symengine.py
sudo python setup.py install