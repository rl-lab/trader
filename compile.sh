# apt install gcc g++ make cmake build-essential python3 python3-pip vim -y

#  add to ~/.bashrc
# export CUDA_HOME="/usr/local/cuda-11.6"
# export PATH="$PATH:/usr/local/cuda/bin"
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"


rm -rf CMakeCache.txt  CMakeFiles/ build/ && mkdir build && cd build &&  cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` .. && make -j4 && cd -
