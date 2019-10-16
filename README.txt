TL;DR: This is a mini-app currently under development which is investigating
strategies for computing many (~100k) small matrix-matrix multiplication
operations in a massively parallel framework.

Short term problem:
    Many quantum many-body physics codes rely on massively parallel tensor
contractions. To get performance, a typical strategy is to permute the tensors
so that they can be re-written as matrix-matrix multiplication. These matrices
are extremely sparse with a known pattern dictated by the physical symmetries of
the target problem. By exploiting these symmetries, a large sparse matrix can be
stored as a block diagonal matrix with ~100k blocks of a wide range of sizes
(from 1 element to a trillion). Handling of the large matrices is ok, but the
performance of computing thousands of small gemms is very poor right now.
Looping over thousands of small cublas calls causes performance to be bogged
down in overhead. Currently looping over host side openblas gemm calls is
faster.
    This mini-app is investigating any solutions that might improve the
performance of computing many small gemms on GPUs. Currently the only other
test implemented is using the magma library magma_dgemm_vbatched call, which
seems to have very good performance, but breaks (gives 0 for A*B) for large
(>2^15 or so) numbers of matrices in the batch.

Medium term problem:
    A library call which computes the full tensor contraction would be ideal
long term. It is common to see terms in many-body theory which might look like
(using Einstein summation notation)

T^{ab}_{ij} = alpha * V^{ab}_{cd} * T^{cd}_{ij} + beta * T^{ab}_{ij}

where in general, alpha and beta are complex numbers (where single precision is
typically insufficient) as are the entries of the rank 4 (or higher) tensors A,
B, and C.  Each of these tensor indices can represent thousands or tens of
thousands of states, so naively storing these objects as 4-indexed arrays
T[a][b][i][j] woudl require about

(10,000)^4 *(16 bytes) = 160 Pb

of memory each. However, the sparse structure can be exploited by creating a new
2-(or higher) body basis. By chosing the 2-body basis (a,b) -> A in a way that
maps onto physical symmetries, we can rewrite the tensors as a block diagonal
structure

T[a][b][i][j] -> T'[block][A][I]

where by storing only this large array of matrices on the diagonal, we can
reduce the memory requirements by a factor of ~10,000 making these calculations
feasible with distributed memory schemes. This means that the tensor contraction
expressed above would now be computed as

for iBlock in Blocks
T[iBlock]^{A}_{I} = alpha*V[iBlock]^{A}_{C} * T[iBlock]^{C}_{I}
                    + beta*T[iBlock]^{A}_{I}

which is a large loop over gemm operations. Since the tensor contractions over
the 1-body basis indices (c,d) are mapped nicely to a tensor contraction
(matmult) over the 2-body basis index (C).



Long term problem:
There are many terms which needs tensor permutations along with gemm operations,
but we are just looking at the gemm portion of the kernel in this mini-app for
now.
