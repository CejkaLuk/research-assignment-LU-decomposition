# Parallel LU Decomposition for the GPU

This CD contains the PDF, the Decomposition project and other files that are related to the bachelor project _Parallel LU Decomposition for the GPU_.

## Contents
1. PDF of the Research assignment
2. Decomposition project installation
	1. Prerequisites
	2. Dependencies
	3. GitLab project
	4. How to install
	5. How to use
3. Full benchmark results

## 1. PDF of the Research assignment
The PDF is located in the main directory of this CD.
## 2. Decomposition project installation
### i. Prerequisites

* CUDA-compatible graphics card with compute capability of least 6.0 or greater (Kepler architecture or newer).
* 8GB of RAM.
* 4GB of free disk space.
* Linux-based operating system. This guide was done with Ubuntu 18.04 or later in mind.

### ii. Dependencies
In order to install the Decomposition project fully, some dependencies need to be installed first.

* Git - to clone the TNL/Decomposition repository.
```bash
sudo apt update
sudo apt install git
```
* CMake 3.20.5 or later - to compile the project.
```bash
wget https://github.com/Kitware/CMake/releases/download/v3.20.5/cmake-3.20.5.tar.gz
tar -zxvf cmake-3.20.5.tar.gz
cd cmake-3.20.5
./bootstrap
make
sudo make install
```
* GCC 8.4.0 or later (or any compiler that supports the C++14 standard) - to compile the contents of the project.
```bash
sudo apt update
sudo apt install gcc
```
* CUDA 11.5 or later - to compile the CUDA part of the project. The installation guide can be found on [NVIDIA's documentation website](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
* Python 3.9 or later (__OPTIONAL__, but recommended) - this is only if you'd like to use bindings and scripts located in `lu-decomposition/src/Benchmark/scripts/`.
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python 3.9
```
* TNL library - it is a dependency for the project. The installation procedure can be found on the [project's GitLab repository](https://gitlab.com/tnl-project/tnl). To download and install the latest stable version, run the following commands:
```bash
git clone https://gitlab.com/tnl-project/tnl.git
cd tnl
# If have Python installed, then use:
./install --install=yes all
# If you do not have Python installed, then you must let the install script know like so:
./install --with-python=no --install=yes all
```

### iii. GitLab project
The readme to the Decomposition project can be found on the [project's GitLab repository](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/lu-decomposition/).
If you would like to download the latest stable version of the project, then you can clone the repository via HTTPS:
```bash
git clone https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/lu-decomposition.git
```
Otherwise, if you'd like the same version that the benchmarks were run on, please see the `lu-decomposition` directory in this CD.

### iv. How to install
There is one simple command that installs the entire Decomposition project. Inside the `lu-decomposition` directory run the following command:
```bash
./install --install=yes all
```
If you would like more information about the different options this command provides, use:
```bash
./install --help
```

### v. How to use
* To run the unit tests:
	* Go into `lu-decomposition/Debug/bin` and run the individual scripts to run unit tests for individual formats:
* To run the benchmark for a specific matrix:
	* Launch the benchmarking script in the directory of the mtx file with the mtx file as an input parameter:
	```bash
	~/${PATH_TO_DECOMPOSITION}/Debug/bin/decomposition-benchmark-dbg --input-file ${mtx_file_name}.mtx
	```
	* The benchmarks will produce a log that can be parsed to HTML using a python script located in `lu-decomposition/src/Benchmark/scripts/`. Use the script in the directory with the log file and pass the log file as an input parameter:
	```bash
	Python3.9 ~/${PATH_TO_DECOMPOSITION}/lu-decomposition/src/Benchmark/scripts/decomposition-benchmark-make-tables-json.py --input_file ${log_file_name}.log
	```
* To run the benchmark across multiple matrices:
	* Use the script `lu-decomposition/src/Benchmark/scripts/run-decomposition-benchmark`. However, this script requires to be launched in the directory that contains:
		The directory, called `matrices`, has to contain the unpacked `*.mtx` files of the matrices that the benchmark is to be run on. For example:
		```
		matrices/HB/bcsstk03.mtx
		matrices/MKS/fp.mtx
		matrices/Cejka10793.mtx
		```
	* To summarize, run the script:
	```bash
	~/${PATH_TO_DECOMPOSITION}/lu-decomposition/src/Benchmark/scripts/run-decomposition-benchmark
	```
	in the directory that contains:
	```bash
	# Directory which contains the .mtx files of matrices.
	matrices/
	```
	* The matrices that the benchmarks were run on are __available on demand__ as they total around 20GB, therefore could not fit on this CD.

## 3. Full benchmark results
The directory `benchmarks` on this CD contains directories:

* `set_14_matrices` - Full results for the subset of 14 analyzed matrices parsed into excel.
* `set_63_matrices` - Full results for the set of 63 matrices parsed into excel.
* `rci_cluster_results` - Full logs of the benchmarks along with the transformed HTMLs and graphs in PDFs.