%% Helper: Project to line Ax = b
function x_proj = project_to_line(x, A, b)
    A = A(:);     % Ensure column
    x = x(:);     % Ensure column
    alpha = (A' * x - b) / (A' * A);  % Scalar projection
    x_proj = x - alpha * A;
end