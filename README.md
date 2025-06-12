# G-CoMVKM: Globally Collaborative Multi-View k-Means Clustering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-MDPI%20Electronics-blue)](https://www.mdpi.com/2079-9292/14/11/2129)

This repository contains the implementation of the Globally Collaborative Multi-View k-Means (G-CoMVKM) clustering algorithm, as described in our paper published in Electronics journal.

## Overview

G-CoMVKM is a novel clustering algorithm designed for multi-view data. It integrates a collaborative transfer learning framework with entropy-regularized feature-view reduction, enabling dynamic elimination of uninformative components. The method achieves clustering by balancing local view importance and global consensus, without relying on matrix reconstruction.

### Key Features

- **Feature-View Reduction**: Automatically identifies and eliminates uninformative features
- **Adaptive View Weighting**: Dynamically determines the importance of each view
- **Global Consensus**: Balances local view information with global consensus
- **No Matrix Reconstruction**: Avoids computationally expensive matrix reconstruction operations

## Installation

1. Clone this repository to your local machine
2. Run the initialization script to set up the paths:
```matlab
run_me_first
```

## Usage

### Quick Start

```matlab
% Load your multi-view data
% X should be a cell array where each cell contains a data view
% Example: X{1} is the first view, X{2} is the second view, etc.

% Set options
options = struct();
options.clusters = 2;       % Number of clusters
options.gamma = 0.5;        % Parameter for view weights (0-∞)
options.theta = 0.1;        % Parameter for feature weights (>0)
options.visualize = true;   % Show visualization

% Run G-CoMVKM
[X_new, U, A, V, W, delta, dh] = run_G_CoMVKM(X, options);

% Display results
disp('Final view weights:');
disp(V);
disp('Final dimensions per view:');
disp(dh);
```

### Demo

Run the demonstration script to see G-CoMVKM in action with synthetic data:

```matlab
demo_G_CoMVKM_2V2D2C
```

### Advanced Usage

For more detailed information, refer to the comprehensive guide:

```matlab
help G_CoMVKM_guide
```

## Parameters

G-CoMVKM has two key parameters:

1. **Gamma (γ)**: Controls the balance between local view information and global consensus.
   - Range: [0, 1]
   - Values closer to 0: More emphasis on local view information
   - Values closer to 1: More emphasis on global consensus

2. **Theta (θ)**: Controls the feature weighting and dimensionality reduction.
   - Range: (0, ∞)
   - Smaller values: More aggressive feature reduction
   - Larger values: More features are retained

## Output Interpretation

After running G-CoMVKM, you will obtain several outputs:

- **X_new**: The reduced dimensionality dataset
- **U**: The final cluster memberships for each view
- **A**: The final cluster centers for each view
- **V**: The final view weights - higher values indicate more important views
- **W**: The final feature weights for each view
- **delta**: The final regularization parameter
- **dh**: The final number of dimensions for each view

## Files in this Repository

- `G_CoMVKM.m`: Main algorithm implementation
- `run_G_CoMVKM.m`: Wrapper function with parameter validation and visualization
- `demo_G_CoMVKM_2V2D2C.m`: Demonstration script with synthetic data
- `G_CoMVKM_guide.m`: Comprehensive guide with usage examples
- `run_me_first.m`: Initialization script to set up paths

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@Article{electronics14112129,
AUTHOR = {Sinaga, Kristina P. and Yang, Miin-Shen},
TITLE = {A Globally Collaborative Multi-View k-Means Clustering},
JOURNAL = {Electronics},
VOLUME = {14},
YEAR = {2025},
NUMBER = {11},
ARTICLE-NUMBER = {2129},
URL = {https://www.mdpi.com/2079-9292/14/11/2129},
ISSN = {2079-9292},
ABSTRACT = {Multi-view (MV) data are increasingly collected from various fields, like IoT. The surge in MV data demands clustering algorithms capable of handling heterogeneous features and high dimensionality. Existing feature-weighted MV k-means (MVKM) algorithms often neglect effective dimensionality reduction such that their scalability and interpretability are limited. To address this, we propose a novel procedure for clustering MV data, namely a globally collaborative MVKM (G-CoMVKM) clustering algorithm. The proposed G-CoMVKM integrates a collaborative transfer learning framework with entropy-regularized feature-view reduction, enabling dynamic elimination of uninformative components. This method achieves clustering by balancing local view importance and global consensus, without relying on matrix reconstruction. We design a feature-view reduction by embedding transferred learning processes across view components by using penalty terms and entropy to simultaneously reduce these unimportant feature-view components. Experiments on synthetic and real-world datasets demonstrate that G-CoMVKM consistently outperforms these existing MVKM clustering algorithms in clustering accuracy, performance, and dimensionality reduction, affirming its robustness and efficiency.},
DOI = {10.3390/electronics14112129}
}
```

### 💫 Beyond the "Impossible"

As Arthur C. Clarke said, "The only way of discovering the limits of the possible is to venture a little way past them into the impossible."

We didn't just venture—we blazed a trail:

- Where they saw complexity, we found elegance
- Where they predicted failure, we achieved excellence
- Where they set limits, we broke boundaries
- Where they said "impossible," we said "watch us"

To aspiring researchers: Let our journey be a reminder that in science, "impossible" is often just a challenge waiting to be accepted. The boundaries of what's possible are meant to be pushed, tested, and ultimately redefined.

## Contact

- **Kristina P. Sinaga**
- Email: kristinasinaga41@gmail.com
