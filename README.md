# GPU Programming Concepts <br />

This repository contains the material corresponding to the webinar/Hands-on: _GPU Programming Concepts_ . The objectives are:

1. **Present** quickly and objectively concepts of Code Portability and Optimization;
2. **Show** through simple examples the use of Parallel and Distributed Programming in multi-GPU systems;
3. **Expose** students to applicability in case studies.

Material available in \*.ipynb (Jupyter NoteBook) format. There are 8 notebooks in this repository:

## [1-Introduction.ipynb](https://drive.google.com/file/d/1cVo4tyMFsdYi4I-Yf5YllVh4GF2TFYuv/view?usp=sharing)
The introductory notebook contains some of the webinar's content, and through some basic commands, we will be able to characterize the available multi-GPU execution environment.

## [2-NCCL-P2P.ipynb](https://drive.google.com/file/d/1GS-kuYEeJZJ_FJ5niAA-pmFnMn7SBOzS/view?usp=sharing)
This notebook will introduce peer-to-peer direct memory access on GPUs through the NCCL API.

## [3-CUDAWARE-MPI.ipynb](https://drive.google.com/file/d/1-_3lWEsOEmhUaSj4rvcfY-ARxsi1Hzvv/view?usp=sharing)
In this notebook, we will introduce and present the features of MPI and CUDA compatibility and explain how it is efficient and can be maximized in the CUDA-aware MPI API.

## [4-MCπ-SGPU.ipynb](https://drive.google.com/file/d/1pP6WJsTdLXfz8fcBYumFKuVt_ce18ljg/view?usp=sharing)
We will use the highly parallelizable algorithm (Monte Carlo Approximation for the Calculus of π) to discuss multi-GPU programming. In this Notebook, we'll introduce the algorithm and start our exploration by running it on a single GPU.

## [5-MCπ-MGPU.ipynb](https://drive.google.com/file/d/11kk9C2kwvzcKtsTu1BmSSGYIdeS2PE63/view?usp=sharing)
In this notebook, we will refactor the single-GPU implementation of the Monte Carlo approximation of the π algorithm to run on multiple GPUs. While this is a valid refactoring technique, we hope to begin demonstrating the porting and optimization process through the CUDA-aware MPI and NVSHMEM APIs.

## [6-MCπ-CUDAWARE-MPI.ipynb](https://drive.google.com/file/d/1QEZgEP_8dB5pPuScXyqPBNLNQ_qWa9ni/view?usp=sharing)
In this notebook, we will deepen the concepts of the CUDA-aware MPI API, which will grant us the benefits of the SPMD paradigm, maintaining the ability to use memory point-to-point over multi-GPU environments.

## [7-MCπ-NVSHMEM.ipynb](https://drive.google.com/file/d/1xCflx552h5QiSVOWmN40GHXuIFYaeC-v/view?usp=sharing)
In this notebook, we will introduce NVSHMEM and make a first attempt to use it in the Monte Carlo approximation of the π program.

## [8-Jacobi.ipynb](https://drive.google.com/file/d/1BbSHsvtI9EtlTFRgTdHs5-edSrpt0b7c/view?usp=sharing)
We finish with the solution of the Laplace equations using the Jacobi iteration. This problem will allow us to explore the common motive of dealing with distributed data communication.

---

[Murilo Boratto](http://lattes.cnpq.br/9222855062709254) <br/>
Researcher at the Supercomputing Center for Industrial Innovation SENAI-CIMATEC [CS2I](http://www.senaicimatec.com.br/) <br/>
