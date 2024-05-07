################################################################
#   C. Jarne 2023 Analysis Group of D. Vidahurre  @cfin        #
# The code generates N sets of histograms (sample_set_size)    #
# from gamma distributions of two classes and labels them.     #
# Each set has the same number of samples called "sizes"       #
# and is saved in the "synthetic-data-grid" directory with     #
# a subdirectory that indicates samples and the difference     #
# between the alpha values. We can create as many different    #
# sets as desired will be created to study the effect on the   #
# number of samples in the histogram or the effect of the      #
# difference in parameters.                                    #
################################################################
import numpy as np
from scipy.stats import gamma
import os

# Definition of the parameters for the gamma distributions
alpha1 = 5  # Alpha parameter of the first distribution
beta1 = 10  # Beta parameter of the first distribution
beta2 = 10  # Beta parameter of the second distribution
sizes = [50, 100, 200, 300, 400, 500]  # Different sizes for the distributions
# sizes = [600, 700, 800, 900]
x = np.linspace(0, 49, 49)
alpha_diff = [0.25]
# alpha_diff = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2] # 1%, 2%....
sample_set_size = 200  # how many distributions of each class will be generated

# Create the directory if it doesn't exist
directory = "synthetic-data-grid"
if not os.path.exists(directory):
    os.makedirs(directory)

# Iterate over different alpha2 values and sizes
for i in alpha_diff:
    # Different alpha2 values
    alpha2 = alpha1 - i * alpha1  # alpha 0.01, 0.02, 0.05, 0.1, 0.15, 0.2
    print(alpha2)
    for size in sizes:
        # Create directory for this combination of parameters
        subdir = f"{directory}/alpha2_{i}_size_{size}"
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        # Iterate 100 times to create 100 data sets
        for j in range(sample_set_size):
            # Generate random numbers for each distribution
            numbers1 = gamma.rvs(alpha1, beta1, size=size)
            numbers2 = gamma.rvs(alpha2, beta2, size=size)

            # Save the numbers in text files
            np.savetxt(f"{subdir}/sample_{j+1}_young.txt", numbers1)
            np.savetxt(f"{subdir}/sample_{j+1}_old.txt", numbers2)
