function [assignment, cost] = munkres(costMatrix)
% MUNKRES - Hungarian algorithm/Munkres algorithm implementation
%
% This function implements the Hungarian algorithm (also known as the Munkres
% algorithm) for solving the assignment problem. It finds the optimal
% assignment of rows to columns in a cost matrix to minimize the total cost.
%
% Inputs:
%   costMatrix - Cost matrix (n x n)
%
% Outputs:
%   assignment - Optimal assignment (n x 1)
%   cost       - Total cost of the assignment
%
% NOTE: This is a simplified implementation for the G-CoMVKM algorithm.
% If you have the Optimization Toolbox, you can use MATLAB's built-in
% assignmentoptimal function for better performance.

% Check if the optimization toolbox is available and has the assignmentoptimal function
if exist('assignmentoptimal', 'file') == 2
    % Use MATLAB's built-in function
    [assignment, cost] = assignmentoptimal(costMatrix);
    return;
end

% Get the size of the cost matrix
[n, m] = size(costMatrix);

% Make sure the cost matrix is square
if n ~= m
    error('Cost matrix must be square');
end

% Step 1: For each row, subtract the minimum value from all elements
costMatrix = costMatrix - min(costMatrix, [], 2);

% Step 2: For each column, subtract the minimum value from all elements
costMatrix = costMatrix - min(costMatrix, [], 1);

% Step 3: Cover all zeros with minimum number of lines
covered_rows = false(n, 1);
covered_cols = false(1, n);
assignment = zeros(n, 1);

% Find initial assignment
for i = 1:n
    for j = 1:n
        if costMatrix(i, j) == 0 && ~covered_rows(i) && ~covered_cols(j)
            assignment(i) = j;
            covered_rows(i) = true;
            covered_cols(j) = true;
            break;
        end
    end
end

% If all rows are covered, we're done
while sum(assignment > 0) < n
    % Find a zero that is not covered
    zero_found = false;
    for i = 1:n
        if ~covered_rows(i)
            for j = 1:n
                if costMatrix(i, j) == 0 && ~covered_cols(j)
                    % Mark this zero
                    prime_row = i;
                    prime_col = j;
                    zero_found = true;
                    break;
                end
            end
            if zero_found
                break;
            end
        end
    end
    
    if ~zero_found
        % No uncovered zero found, create new zeros
        % Find the minimum uncovered value
        min_val = inf;
        for i = 1:n
            if ~covered_rows(i)
                for j = 1:n
                    if ~covered_cols(j) && costMatrix(i, j) < min_val
                        min_val = costMatrix(i, j);
                    end
                end
            end
        end
        
        % Subtract min_val from all uncovered rows
        for i = 1:n
            if ~covered_rows(i)
                costMatrix(i, :) = costMatrix(i, :) - min_val;
            end
        end
        
        % Add min_val to all covered columns
        for j = 1:n
            if covered_cols(j)
                costMatrix(:, j) = costMatrix(:, j) + min_val;
            end
        end
        continue;
    end
    
    % Check if there is a starred zero in the row
    star_col = find(assignment(prime_row));
    if isempty(star_col)
        % No starred zero, augment path and reset
        augment_path = [prime_row, prime_col];
        while prime_col > 0
            % Find starred zero in this column
            star_row = find(assignment == prime_col);
            if isempty(star_row)
                break;
            end
            star_row = star_row(1);
            augment_path = [augment_path; star_row, prime_col];
            
            % Find primed zero in this row
            prime_col = find(costMatrix(star_row, :) == 0 & ~covered_cols, 1);
            if isempty(prime_col)
                break;
            end
            augment_path = [augment_path; star_row, prime_col];
        end
        
        % Augment path: unstar each starred zero, star each primed zero
        for i = 1:size(augment_path, 1)
            if mod(i, 2) == 1
                assignment(augment_path(i, 1)) = augment_path(i, 2);
            else
                assignment(augment_path(i, 1)) = 0;
            end
        end
        
        % Reset covers
        covered_rows = false(n, 1);
        covered_cols = false(1, n);
        for i = 1:n
            if assignment(i) > 0
                covered_rows(i) = true;
                covered_cols(assignment(i)) = true;
            end
        end
    else
        % There is a starred zero in the row, cover the row and uncover the column
        covered_rows(prime_row) = true;
        covered_cols(star_col) = false;
    end
end

% Calculate the total cost
cost = 0;
for i = 1:n
    if assignment(i) > 0
        cost = cost + costMatrix(i, assignment(i));
    end
end
end
