## parameters
M = 500  # Number of hypotheses for RANSAC
thr = 0.1  # RANSAC threshold
C1 = 10  # Resolution/grid-size for the mapping function in MDLT (C1 x C2).
C2 = 10 # 一开始是50,计算的太慢了,调成10快了很多
gamma = 0.01  # Normalizer for Moving DLT. (0.0015-0.1 are usually good numbers).
sigma = 8.5  # Bandwidth for Moving DLT. (Between 8-12 are good numbers).
