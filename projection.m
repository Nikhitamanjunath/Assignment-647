%% Helper: Projection function
% Constraint: minimize f0(x1, x2)
% Subject to:
%   2x1 + x2 >= 3
%   x1 + 2x2 >= 3
%   x1, x2 >= 0
function x_proj = projection(x)
    x_proj = max(x(:), 0); % Ensure column and non-negativity
    % Project to satisfy constraints iteratively
    max_iter_proj = 5;
    for i = 1:max_iter_proj
        if 2*x_proj(1) + x_proj(2) < 3
            A = [2; 1]; b = 3;
            x_proj = project_to_line(x_proj, A, b);
            x_proj = max(x_proj, 0);
        end
        if x_proj(1) + 2*x_proj(2) < 3
            A = [1; 2]; b = 3;
            x_proj = project_to_line(x_proj, A, b);
            x_proj = max(x_proj, 0);
        end
    end
end