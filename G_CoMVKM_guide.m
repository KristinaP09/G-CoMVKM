%% G-CoMVKM Algorithm Guide
% This document provides a comprehensive guide to using the G-CoMVKM algorithm
% for clustering multi-view data.
%
% G-CoMVKM: Globally Collaborative Multi-View k-Means Clustering
% Developed by Kristina P. Sinaga
%
% Reference:
% Sinaga, K. P., & Yang, M.-S. (2025). A Globally Collaborative Multi-View 
% k-Means Clustering. Electronics, 14(11), 2129.
% https://www.mdpi.com/2079-9292/14/11/2129

%% Introduction
%
% Multi-view clustering is a technique for analyzing data that is described by multiple
% feature sets (views). G-CoMVKM integrates a collaborative transfer learning
% framework with entropy-regularized feature-view reduction, enabling dynamic
% elimination of uninformative components. This method achieves clustering by
% balancing local view importance and global consensus.
%
% Key Features of G-CoMVKM:
% 1. Automatically determines the importance of each view
% 2. Eliminates irrelevant features to reduce dimensionality
% 3. Balances local view information with global consensus
% 4. Works well on heterogeneous multi-view data

%% Algorithm Overview
%
% G-CoMVKM iteratively performs the following steps:
%
% 1. Compute membership matrices for each view
% 2. Update cluster centers for each view
% 3. Update feature weights for each view
% 4. Discard irrelevant feature-view components
% 5. Adjust variables based on the reduced feature set
% 6. Update view weights
% 7. Compute the objective function value
% 8. Check for convergence
%
% This process continues until convergence or the maximum number of iterations is reached.

%% Usage Example
%
% The following code demonstrates how to use G-CoMVKM on a sample dataset:

% Generate a simple multi-view dataset (2 views, 3 clusters)
num_samples = 100;
X = cell(1, 2);
% View 1: 5-dimensional data
X{1} = [randn(30, 5) + 3; randn(30, 5) - 3; randn(40, 5)];
% View 2: 8-dimensional data
X{2} = [randn(30, 8) + 2; randn(30, 8) - 2; randn(40, 8)];

% Set options
options = struct();
options.clusters = 3;       % Number of clusters
options.gamma = 0.6;        % Parameter for view weights (0-1)
options.theta = 0.2;        % Parameter for feature weights (>0)
options.maxIter = 50;       % Maximum iterations
options.visualize = true;   % Show visualization
options.thresholdType = 'general'; % Threshold type for feature selection

% Run G-CoMVKM
[X_new, U, A, V, W, delta, dh] = run_G_CoMVKM(X, options);

% Display results
disp('Final view weights:');
disp(V);
disp('Final dimensions per view:');
disp(dh);

%% Parameter Selection
%
% The G-CoMVKM algorithm has two key parameters:
%
% 1. Gamma (γ): Controls the balance between local view information and global consensus.
%    - Range: [0, 1]
%    - Values closer to 0: More emphasis on local view information
%    - Values closer to 1: More emphasis on global consensus
%    - Recommended default: 0.5
%
% 2. Theta (θ): Controls the feature weighting and dimensionality reduction.
%    - Range: (0, ∞)
%    - Smaller values: More aggressive feature reduction
%    - Larger values: More features are retained
%    - Recommended default: 0.1-0.2
%
% Parameter selection tips:
% - For exploratory analysis, start with γ=0.5 and θ=0.1
% - For datasets with highly heterogeneous views, decrease γ
% - For datasets where you want to preserve more features, increase θ
% - If dimensionality reduction is too aggressive, increase θ
% - If views should contribute more equally, increase γ

%% Threshold Selection for Feature Reduction
%
% G-CoMVKM uses thresholds to determine which features to retain:
%
% Available threshold types:
% - 'artificial':   1/sum(dh)                    - Best for artificial datasets
% - 'general':      1/data_n                     - General purpose
% - 'wikipedia':    s./sqrt(data_n*sum(dh))      - Wikipedia articles datasets
% - 'prokaryotic':  time/sqrt(data_n*sum(dh))    - Prokaryotic datasets
% - 'leaves':       Param1*((1)./sqrt(data_n*d_h)) - 100 Leaves Dataset
%
% The choice of threshold affects how aggressively features are reduced.

%% Interpreting Results
%
% After running G-CoMVKM, you will obtain several outputs:
%
% - X_new: The reduced dimensionality dataset
% - U: The final cluster memberships for each view
% - A: The final cluster centers for each view
% - V: The final view weights - higher values indicate more important views
% - W: The final feature weights for each view - higher values indicate more important features
% - delta: The final regularization parameter
% - dh: The final number of dimensions for each view
%
% Key insights from results:
% 1. View importance: Examine V to see which views contribute most to clustering
% 2. Feature reduction: Compare original dimensions with dh to see how many features were eliminated
% 3. Clustering quality: Analyze U to determine how well-separated the clusters are

%% Additional Tips
%
% - Data preprocessing: Normalize/standardize features before using G-CoMVKM
% - Initialization: The algorithm is sensitive to initialization; consider running multiple times
% - Large datasets: For very large datasets, consider sampling or incremental processing
% - Visualization: Use t-SNE or PCA to visualize the clustering results for high-dimensional data
% - Evaluation: Use external metrics (if ground truth is available) or internal metrics (like silhouette) to evaluate quality
%
% For more details, refer to the original paper or contact the author.
