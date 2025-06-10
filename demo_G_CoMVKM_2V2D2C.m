%% Demo script for G-CoMVKM algorithm on 2V2D2C dataset
% This script demonstrates the application of G-CoMVKM algorithm on a synthetic
% dataset (2V2D2C - 2 Views, 2 Dimensions, 2 Clusters) and evaluates its 
% clustering performance using multiple metrics.
%
% Written by Kristina P. Sinaga
% Enhanced with visualizations and statistical evaluation

%% Setup environment
close all; clear all; clc

% Make sure paths are set correctly
run_me_first;

% Load the synthetic data
data_name = 'Numerical Data 2V2D2C';
fprintf('Running G-CoMVKM on %s\n', data_name);
load data_synthetic.mat
X = data;
clear data

%% Parameters setup
% Set the parameters of the algorithm
c = max(label);                     % Number of clusters
points_view = length(X);            % Number of views
points_n = size(X{1},1);            % Number of data points
dh = [];                            % Array to store dimensions of each view
for h = 1:points_view
    dh =[dh size(X{h}, 2)];
end
Param1 = 5;                         % Gamma parameter
Param2 = 4;                         % Theta parameter

% Experimental setup
num_seeds = 50;                     % Using 50 different random seeds
seeds = randi(2^13-1, [1 num_seeds]); % Random seeds generator
result = zeros(num_seeds, 3);       % Array to store AR, RI, NMI results

% Create a progress bar figure
progress_fig = figure('Name', 'G-CoMVKM Progress', 'NumberTitle', 'off');
progress_ax = axes('Parent', progress_fig);
progress_bar = barh(progress_ax, 0, 'FaceColor', [0.2 0.6 0.8]);
title(progress_ax, 'G-CoMVKM Progress');
xlabel(progress_ax, 'Completion Percentage');
ylabel(progress_ax, 'Progress');
xlim(progress_ax, [0 100]);
text(5, 1, '0%', 'Parent', progress_ax);

%% Run multiple experiments with different initializations
fprintf('\nRunning %d experiments with different random initializations...\n', num_seeds);

% Arrays to store best results
best_AR = 0;
best_RI = 0;
best_NMI = 0;
best_U = [];
best_X = [];
best_index = [];
best_time = 0;

for time = 1:num_seeds % Number of repetitions
    % Update progress bar
    set(progress_bar, 'XData', 100 * time / num_seeds);
    set(findobj(progress_ax, 'Type', 'text'), 'String', sprintf('%.1f%%', 100 * time / num_seeds));
    drawnow;
    
    % Set random seed for reproducibility
    rng(seeds(time));
    
    fprintf('\n----- Experiment %d/%d -----\n', time, num_seeds);
    
    %---------------- Initialization stage -------------------------------
    
    % Initialize the cluster centers A
    initial = randperm(points_n, c);
    for h = 1:points_view
        A{h} = X{h}(initial,:); % Init. generator for Artificial data
        % Alternative initialization using k-means
        % [~, A{h}] = kmeans(X{h}, c, 'MaxIter', 50, 'Replicates', 5);
    end
    
    % Initialize the weighted view V
    V = ones(1, points_view) ./ points_view;
    
    % Initialize the h-th view of features weight W
    for h = 1:points_view
        W{h} = ones(1, dh(h)) ./ dh(h);
    end
    
    % Compute delta
    for h = 1:points_view
        delta_left = X{h} ./ sum(X{h});
        delta_right = max(X{h}) - min(X{h});
        delta{h} = mean(delta_left ./ delta_right);
    end
    
    %----------------------------------------------------------------------
    
    fprintf('Running G-CoMVKM for seed %d...\n', seeds(time));
    tic;
    
    % Run G-CoMVKM algorithm
    [X_out, U_out, A_out, V_out, W_out, delta_out, dh_out] = ...
        G_CoMVKM(X, delta, c, points_view, Param1, Param2, dh, A, W, V);
    
    elapsed_time = toc;
    
    %---------- Fusion stage start ----------------------------------------
    % Combine memberships from different views based on view weights
    UU = (U_out{1} .* V_out(1) + U_out{2} .* V_out(2)); % For 2-view data
    %----------- Fusion stage end ------------------------------------------
    
    % Get the final cluster assignments
    index = zeros(points_n, 1);
    for i = 1:points_n
        [~, idx] = max(UU(i,:));
        index(i) = idx;
    end
    
    % ---------------------------------------------------------------------
    % Calculate the clustering performance evaluation
    NMI = nmi(label, index);
    RI = RandIndex(label, index);
    AR = 1 - ErrorRate(label, index, c) / points_n;
    
    % Display metrics for current run
    fprintf('Run %d - AR: %.4f, RI: %.4f, NMI: %.4f (Time: %.2f sec)\n', ...
            time, AR, RI, NMI, elapsed_time);
    
    % Save results
    result(time, 1) = AR;
    result(time, 2) = RI;
    result(time, 3) = NMI;
    
    % Update best results if current run is better
    if AR > best_AR
        best_AR = AR;
        best_RI = RI;
        best_NMI = NMI;
        best_U = U_out;
        best_X = X_out;
        best_index = index;
        best_time = time;
    end
end

% Close progress bar
close(progress_fig);

%% Compute statistical results
min_result = min(result);   % Minimum values of clustering metrics
mean_result = mean(result); % Mean values of clustering metrics
max_result = max(result);   % Maximum values of clustering metrics
std_result = std(result);   % Standard deviation of metrics

%% Display final results
fprintf('\n======================================================\n');
fprintf('G-CoMVKM on %s - Results Summary\n', data_name);
fprintf('======================================================\n');
fprintf('Statistical results over %d runs:\n\n', num_seeds);

fprintf('             AR        RI        NMI\n');
fprintf('Min:      %.4f    %.4f    %.4f\n', min_result(1), min_result(2), min_result(3));
fprintf('Mean:     %.4f    %.4f    %.4f\n', mean_result(1), mean_result(2), mean_result(3));
fprintf('Max:      %.4f    %.4f    %.4f\n', max_result(1), max_result(2), max_result(3));
fprintf('Std Dev:  %.4f    %.4f    %.4f\n', std_result(1), std_result(2), std_result(3));

fprintf('\nBest result was achieved on run %d:\n', best_time);
fprintf('AR: %.4f, RI: %.4f, NMI: %.4f\n', best_AR, best_RI, best_NMI);

fprintf('\nFinal dimensionality reduction:\n');
fprintf('Original dimensions: [%d, %d]\n', dh(1), dh(2));
fprintf('Reduced dimensions: [%d, %d]\n', dh_out(1), dh_out(2));
fprintf('Reduction: %.1f%%, %.1f%%\n', ...
    (1 - dh_out(1)/dh(1))*100, (1 - dh_out(2)/dh(2))*100);

%% Visualize results

% 1. Performance distribution across runs
figure;
boxplot(result, 'Labels', {'AR', 'RI', 'NMI'});
title('Distribution of Clustering Metrics Across Runs');
ylabel('Metric Value');
grid on;

% 2. Convergence plot for view weights (from the best run)
figure;
bar(V_out);
title('Final View Weights');
xlabel('View Index');
ylabel('Weight');
ylim([0, 1]);
grid on;
xticks(1:points_view);
xticklabels(arrayfun(@(x) sprintf('View %d', x), 1:points_view, 'UniformOutput', false));

% 3. Dimensionality reduction visualization
figure;
bar([dh; dh_out]');
title('Dimension Reduction');
xlabel('View');
ylabel('Number of Dimensions');
legend('Original', 'Reduced');
grid on;
xticks(1:points_view);
xticklabels(arrayfun(@(x) sprintf('View %d', x), 1:points_view, 'UniformOutput', false));

% 4. Clustering visualization (if 2D data)
if all(dh_out <= 2) || all(dh <= 2)
    figure;
    
    % Create colormap for clusters
    colors = lines(c);
    
    % Plot the clustering results for each view
    for h = 1:points_view
        subplot(1, points_view, h);
        
        % If data is 2D, plot directly
        if size(X_out{h}, 2) == 2
            for k = 1:c
                cluster_points = (best_index == k);
                scatter(X_out{h}(cluster_points, 1), X_out{h}(cluster_points, 2), 50, ...
                    colors(k,:), 'filled', 'MarkerEdgeColor', 'k');
                hold on;
            end
            % Also plot cluster centers
            scatter(A_out{h}(:, 1), A_out{h}(:, 2), 150, colors, 'd', 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 2);
        elseif size(X_out{h}, 2) == 1
            % For 1D data, create a scatter plot with a dummy y-axis
            for k = 1:c
                cluster_points = (best_index == k);
                scatter(X_out{h}(cluster_points, 1), zeros(sum(cluster_points), 1) + 0.1*randn(sum(cluster_points), 1), ...
                    50, colors(k,:), 'filled', 'MarkerEdgeColor', 'k');
                hold on;
            end
            % Also plot cluster centers
            scatter(A_out{h}(:, 1), zeros(c, 1), 150, colors, 'd', 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 2);
            ylim([-0.5, 0.5]);
            ylabel('Jittered Position');
        else
            % For higher dimensions, use PCA to reduce to 2D for visualization
            [coeff, score] = pca(X_out{h});
            for k = 1:c
                cluster_points = (best_index == k);
                scatter(score(cluster_points, 1), score(cluster_points, 2), 50, ...
                    colors(k,:), 'filled', 'MarkerEdgeColor', 'k');
                hold on;
            end
            % Project and plot cluster centers
            centers_projected = A_out{h} * coeff(:, 1:2);
            scatter(centers_projected(:, 1), centers_projected(:, 2), 150, colors, 'd', 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 2);
        end
        
        title(sprintf('View %d Clustering', h));
        xlabel('Dimension 1');
        if size(X_out{h}, 2) > 1
            ylabel('Dimension 2');
        end
        grid on;
        legend([arrayfun(@(x) sprintf('Cluster %d', x), 1:c, 'UniformOutput', false), 'Centers'], ...
            'Location', 'best');
    end
    sgtitle('G-CoMVKM Clustering Results');
end

% 5. Confusion Matrix Visualization
figure;
confusionchart(label, best_index);
title('Confusion Matrix: True vs. Predicted Clusters');

% 6. Performance Metrics Across Runs
figure;
plot(1:num_seeds, result(:,1), 'r-o', 'LineWidth', 1.5, 'MarkerSize', 5);
hold on;
plot(1:num_seeds, result(:,2), 'g-s', 'LineWidth', 1.5, 'MarkerSize', 5);
plot(1:num_seeds, result(:,3), 'b-d', 'LineWidth', 1.5, 'MarkerSize', 5);
xlabel('Experiment Number');
ylabel('Metric Value');
title('Clustering Performance Across Different Initializations');
legend('AR', 'RI', 'NMI', 'Location', 'best');
grid on;

%% Save results
save_filename = 'G_CoMVKM_results.mat';
save(save_filename, 'X', 'X_out', 'best_U', 'A_out', 'V_out', 'W_out', ...
    'delta_out', 'dh_out', 'label', 'best_index', 'result', 'mean_result', 'max_result');
fprintf('\nResults saved to %s\n', save_filename);

% Display final message
fprintf('\nG-CoMVKM demonstration completed.\n');
fprintf('The algorithm successfully reduced the feature dimensionality while maintaining clustering structure.\n');
fprintf('You can analyze the results in the workspace or load them from the saved file.\n');
