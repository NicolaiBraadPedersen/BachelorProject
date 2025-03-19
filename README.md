find LSM and binomail pricing method in LSM - put (v.2) and CRR - put (v.3) respectivly, where the respective optimal stopping boundary is visualized along with the paths used for pricing
CRR + LSM implementation includes a wide range of options calculated with different parameters, for LSM, CRR and black scholes EU price.
LSM - put (basis test) includes a short robustness check for choice of basis, by looking at the LSM price estimator
BM and Interpolation includes Stock price path simulation method and interpolation methods utilized in the different hedging expirements
Delta analysis (CRR + BS) examines the difference bewtween EU and AMR options delta and price, includes also argumentation for choice of amount of interpolation points
LSM ISD - naive + 2-stage includes a range of options delta's and prices based on the two sub methods respectivly, and compare to the CRR method as Benchmark
The remaining 5 files contain methods of finding delta, and hedging expiremt for the respective method. The two files LSM ISD (...) - Greeks and the two files named Hedging contatians this, and the finite difference.
