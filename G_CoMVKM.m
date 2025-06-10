%%  Matlab Function of A Globally Collaborative Multi-View k-Means Clustering (MDPI on Electronics)
%   Written by Kristina P. Sinaga 
%   We proposed a globally collaborative MVKM (G-CoMVKM) clustering algorithm
%   Tested on Matlab R2020a and later versions
%   Copyright (c) 2025 Kristina P. Sinaga
%   Contact :  kristinasinaga41@gmail.com
%
%   Description:
%   ------------
%   G-CoMVKM integrates a collaborative transfer learning framework with 
%   entropy-regularized feature-view reduction, enabling dynamic elimination 
%   of uninformative components. This method achieves clustering by balancing 
%   local view importance and global consensus.
%
%-------------------------------------------------------------------------------------------------------------------
% Input:
%       X           - Multi-view dataset (cell array where each cell contains a data view)
%       delta       - Regularization parameter for each view (cell array)
%       cluster_num - Number of clusters to form (integer)
%       points_view - Number of data views (integer)
%       Ggamah      - Exponent parameter to control the weights of V (float, typically in range [0,1])
%       Ttetah      - Coefficient parameter to control the weights of W (float, typically > 0)
%       dh          - Vector containing the number of dimensions for each view (vector)
%       A           - Initial cluster centers for each view (cell array)
%       W           - Initial feature weights for each view (cell array)
%       V           - Initial view weights (vector)
%
% Output:
%       X_out       - The new lower dimensionality produced by G-CoMVKM (relevant set of features)
%       U_out       - The final memberships of each data point to clusters
%       A_out       - The final cluster centers for each view
%       V_out       - The final view weights
%       W_out       - The final feature weights for each view
%       delta_out   - The final regulator parameter
%       dh_out      - The final number of dimensions for each view
%
% Example:
%       [X_new, U, A, V, W, delta, dh] = G_CoMVKM(X, delta, 3, 2, 0.5, 0.1, dh, A, W, V);
%
%-------------------------------------------------------------------------------------------------------------------
 function [ X_out, U_out, A_out, V_out, W_out, delta_out, dh_out ] = G_CoMVKM (X, delta, cluster_num, points_view, Ggamah, Ttetah, dh, A, W, V)
 % Input validation
 if nargin < 10
     error('G-CoMVKM requires 10 input arguments');
 end
 
 if ~iscell(X) || ~iscell(delta) || ~iscell(A) || ~iscell(W)
     error('X, delta, A, and W must be cell arrays');
 end
 
 if ~isnumeric(cluster_num) || cluster_num <= 0 || cluster_num ~= round(cluster_num)
     error('cluster_num must be a positive integer');
 end
 
 if ~isnumeric(points_view) || points_view <= 0 || points_view ~= round(points_view)
     error('points_view must be a positive integer');
 end
 
 if ~isnumeric(Ggamah) || Ggamah < 0 || Ggamah > 1
     warning('Ggamah should typically be in range [0,1]');
 end
 
 if ~isnumeric(Ttetah) || Ttetah <= 0
     warning('Ttetah should typically be a positive value');
 end
 
 % Replacing the parameters with more readable variable names
 s = points_view;           % Number of views
 c = cluster_num;           % Number of clusters
 data_n = size(X{1},1);     % Number of data points
 Param1 = Ggamah;           % Gamma parameter
 Param2 = Ttetah;           % Theta parameter
 
 %-------------------------------------------------------------------------
 % Initializing the variables
 time = 1;                        % Iteration counter
 max_time = 100;                  % Maximum number of iterations
 min_improvement = 1e-4;          % Convergence threshold
 obj_G_CoMVKM = zeros(1,max_time); % Store objective values
 
 % Create figure for visualization of convergence
 figure_handle = figure('Name', 'G-CoMVKM Convergence', 'NumberTitle', 'off');
 convergence_plot = plot(1, 0, 'b-o');
 xlabel('Iteration');
 ylabel('Objective Function Value');
 title('G-CoMVKM Convergence');
 grid on;
 
 % Initialize structs to store intermediate results
 intermediate_results = struct();
 intermediate_results.U = cell(1, max_time);
 intermediate_results.A = cell(1, max_time);
 intermediate_results.W = cell(1, max_time);
 intermediate_results.V = cell(1, max_time);
 
 fprintf('G-CoMVKM: Starting the algorithm with %d views, %d clusters\n', s, c);
 fprintf('G-CoMVKM: Data size: %d samples\n', data_n);
 fprintf('G-CoMVKM: Parameters - Gamma: %.4f, Theta: %.4f\n\n', Param1, Param2);
 
 %----------- Start the iteration -----------------------------------------
 
 while 1 && time <= max_time
     
     % Display clear iteration header
     fprintf('\n============= G-CoMVKM Iteration %d/%d =============\n', time, max_time);
     
     % Optional: Save memory by clearing unnecessary variables between iterations
     if time > 1
         clear WVdel D_new A_sum A_sum_new;
     end 

 %--------- Step 1: compute the memberships U -----------------------------
 
 for h =1:s
     WVdel{h} =W{h}.*delta{h}.*V(h);
     for k = 1:c
         A_sum = repmat(A{h}(k,:),data_n,1);
         D{h}(:,k)  = (X{h} - A_sum).^2*WVdel{h}';              
     end
 end
 
 D_average = zeros(data_n,c);
 for h = 1:s
     D_average = D_average + D{h};
 end
 D_average = (1- Param1*data_n) * D_average;
 
 for h = 1:s
     U{h} = zeros(data_n,c);
     A_dist = (Param1*data_n)*D{h} - D_average;
     [min_dist,Update_cluster_elem]=min(A_dist,[],2);
      Cluster_elem{h} = Update_cluster_elem;
      for i = 1:data_n
          U{h}(i,Cluster_elem{h}(i)) = 1;
      end
 end
 
 %-------- Step 2: updating the cluster centers A -------------------------
 
 U_mix = zeros(data_n,c);
 for h = 1:s
     U_mix = U_mix + U{h};
 end
 
 for h = 1:s
     U_mean{h} = zeros(data_n,c);
 end

 for h = 1:s
     U_mean{h} = (1 - Param1) * U{h} + (Param1) * U_mix;
 end

 for h = 1:s
     for k = 1:c
         A{h}(k,:) = X{h}'* U_mean{h}(:,k) / sum(U_mean{h}(:,k));
     end
 end
 
 %-------- Step 3: update the h-th view of weighted feature W --------------

 for h = 1:s
     B{h}  =zeros(1,dh(h));
 end

 for h = 1:s
     for i = 1:data_n
         for k = 1:c
             if U_mean{h}(i,k) ~=0  
                B{h} = B{h} + ((dh(h)+data_n) / data_n)*V(h)^Param2 * U_mean{h}(i,k)*(X{h}(i,:) -A{h}(k,:)).^2*delta{h}';
             end
         end
     end
     B{h} = 1.\delta{h}.*exp( (-B{h} - Param2) /Param2);
 end

 for h = 1:s
     W{h}(1,:) = B{h}./sum(B{h});
 end
 
 %-------- Step 4: Discard irrelevant feature-view component --------------

 % initialize new dimensionality for each view
 d_new = zeros(1,h);
 d_h_new =[];

 for h = 1:s
     
     % Get the current view
     data_h = X{h};
     
     % storing feature-view components W, delta, A, and d_h
     W_h = W{h};
     delta_h  = delta{h};
     A_h = A{h};
     d_h = dh(h);
     
     % Get the dimensions of the current view
     [data_n, d] = size(data_h);
     
     for j = 1: d
         
         % Set the threshold for feature selection
         % You can uncomment one of these thresholds based on your dataset:
         
         % Common threshold options:
         th_h = 1/sum(dh);                       % Artificial datasets
         % th_h = 1/data_n;                      % General purpose
         % th_h = s./sqrt(data_n*sum(dh));       % Wikipedia articles
         % th_h = time/sqrt(data_n*sum(dh));     % Prokaryotic
         % th_h = Param1*((1)./sqrt(data_n*d_h));% 100 Leaves Data
         
         % Find the unimportant features for the current view 
         exclude_idx = find(W_h < th_h);
           
         %----- Step 5: Adjusting variables W, delta, A, X, and dh --------
         % adjust feature weights W
         W_adj              = W_h;
         W_adj(exclude_idx) = [];
         W_adj              = W_adj/sum(W_adj);

         % adjust delta
         delta_adj              = delta_h;
         delta_adj(exclude_idx) = [];

         % adjust cluster centers A
         A_adj                = A_h;
         A_adj(:,exclude_idx) = [];

         % Adjust the dimensionality dh
         d_h_new = d - length(exclude_idx);

         % Adjust the data-view points X
           data_h_new                = data_h;
           data_h_new(:,exclude_idx) = [];
     end
 
     % Store the new dimensionality after adjustment
     d_new(h) = d_h_new; 
     
     % Store the new MV data after adjustment
     data_temp{h} = data_h_new;     
     
     % Store the new feature-view weights W after adjustmment
     W_h_temp{h} = W_adj;

     % Store the new delta after adjustmment
     delta_h_temp{h} = delta_adj;

     % Store the new cluster centers A after adjustmment 
     A_h_temp{h} = A_adj;
 
 end
 
 W     = W_h_temp;     % W for next step and iteration t+1
 delta = delta_h_temp; % delta for next step and iteration t+1
 A     = A_h_temp;     % A for next step and iteration t+1
 dh    = d_new;        % dh for next step and iteration t+1
 X     = data_temp;    % X for next step and iteration t+1
 
%---------- Step 6: updating the Weighted view V  ------------------------    

% Updating Weighted Euclidean distance D
for h =1:s
    WVdel_new{h} =W{h}.*delta{h};
    for k = 1:c
        A_sum_new = repmat(A{h}(k,:),data_n,1);
        D_new{h}(:,k)  = (X{h} - A_sum_new).^2*WVdel_new{h}';              
    end
end
D = D_new;


V = zeros(1,s);
E = zeros(1,s);
for h = 1:s
    U_mean_right{h} = U_mix - U{h}; 
end

DV_average = zeros(data_n,c);
for h = 1:s
    DV_average = DV_average + D{h};
end
DV_average = (Param1*Param2) * DV_average;

E = zeros(1,s);
for h = 1:s
    Et_left = sum(sum(U_mean{h}.*D{h}));
	Et_right= sum(sum(U_mean_right{h}.*DV_average));
    E(h) = (Et_left+Et_right)^(-1/(Param1-1));
end

for h = 1:s
    V(h) = E(h)/ sum(E);
end 
    
%----------- Computing the objective value  -------------------------------
   
term1 = zeros(s,1);
for h = 1:s
    term1(h) = term1(h) + sum(sum(U{h}.*D{h}));
    term1(h) = term1(h)*V(h).^Param1/dh(h);
end
term1_sum = sum(term1);

term2 = zeros(s,1);
for h = 1:s
    for hh = 1:s
        term2(h) = term2(h) + sum(sum(abs(V(h).^2*(U_mean{h}.*D{h} - U_mean{hh}.*D{hh}))));
    end
end
term2_sum = sum(term2);

term3 = 0;
for h = 1:s
    term3 = term3 + abs(sum(W{h}.*log(delta{h}.*W{h})));
end
term3 = Param2 * term3;

obj_G_CoMVKM(time) = term1_sum + term2_sum + term3;

% Update convergence plot
if time > 1
    set(convergence_plot, 'XData', 1:time, 'YData', obj_G_CoMVKM(1:time));
    drawnow;
end

% Display current objective value and improvement
if time > 1
    improvement = abs(obj_G_CoMVKM(time) - obj_G_CoMVKM(time-1));
    fprintf('G-CoMVKM: Iteration %d, Objective = %.6f, Improvement = %.6f\n', time, obj_G_CoMVKM(time), improvement);
    
    % Store intermediate results for possible analysis
    intermediate_results.U{time} = U;
    intermediate_results.A{time} = A;
    intermediate_results.W{time} = W;
    intermediate_results.V{time} = V;
    
    % Check convergence
    if improvement <= min_improvement
        fprintf('\n==========================================\n');
        fprintf('G-CoMVKM: Converged after %d iterations!\n', time);
        fprintf('Final objective value: %.6f\n', obj_G_CoMVKM(time));
        fprintf('==========================================\n\n');
        break;
    end
else
    fprintf('G-CoMVKM: Iteration %d, Objective = %.6f\n', time, obj_G_CoMVKM(time));
end

% Display summary of feature reduction
fprintf('Feature dimensions after iteration %d: ', time);
fprintf('%d ', dh);
fprintf('\n');

% Display view weights
fprintf('View weights after iteration %d: ', time);
fprintf('%.4f ', V);
fprintf('\n');

time = time + 1;

V_out = V;
X_out = X;
W_out = W;
delta_out = delta;
A_out = A;
dh_out = dh;
U_out = U;

end


