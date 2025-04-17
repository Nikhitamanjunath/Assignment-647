clc; close all;

%% Part 1a
[x1, x2] = meshgrid(-5:0.1:5, -5:0.1:5); 
f = f0(x1,x2); 
figure('Name', '3D Plot');
mesh(x1, x2, f);
title('3D Plot of f(x1, x2)');
xlabel('x1'); ylabel('x2'); zlabel('f(x1, x2)');
directions = [1, 1; -1, 1; 1, -1];
x1_start = 0; x2_start = 0;
figure('Name', 'Function Restricted to Lines');
hold on;
for i = 1:size(directions, 1)
    t = linspace(-5, 5, 100);
    x1_line = x1_start + t * directions(i, 1);
    x2_line = x2_start + t * directions(i, 2);
    f_line = f0(x1_line, x2_line);
    plot(t, f_line, 'LineWidth', 1.5);
end
xlabel('x'); ylabel('f(x)');
title('Function Restricted to Lines');
legend('Direction 1', 'Direction 2', 'Direction 3');
hold off;


%% Part1b
H = [2, 3; 3, 18];
disp('Hessian Matrix H:');
disp(H);

eigVals = eig(H);
disp('Eigenvalues of H:');
disp(eigVals);

if all(eigVals > 0)
    disp('f is strictly convex.');
else
    disp('f is not strictly convex.');
end

syms x1 x2
eq1 = 2*x1 + 3*x2 + 2 == 0;
eq2 = 3*x1 + 18*x2 - 5 == 0;
sol = solve([eq1, eq2], [x1, x2]);
x1_min = double(sol.x1);
x2_min = double(sol.x2);
disp('Global minimizer:');
disp(['x1* = ', num2str(x1_min), ', x2* = ', num2str(x2_min)]);


gamma_small = 0.005; 
gamma_large = 0.20;   
gamma_just  = 0.05;   
gamma_list = [gamma_small, gamma_large, gamma_just];
gamma_labels = {'Small \gamma', 'Large \gamma', 'Just-right \gamma'};

x0 = [0; 0];  
maxIter = 30;  

for i = 1:length(gamma_list)
    gamma = gamma_list(i);
    label = gamma_labels{i};
    [x_hist, f_hist] = run_gradient_descent(x0, gamma, maxIter);
    
    x1_seq = x_hist(1, :);
    x2_seq = x_hist(2, :);
    
    figure('Name', ['Gradient Descent Iterates - ' label]);
    subplot(2,1,1);
    plot(0:maxIter, x1_seq, 'o-', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('x_1');
    title(['x_1 vs Iterations, ' label]);
    
    subplot(2,1,2);
    plot(0:maxIter, x2_seq, 's-', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('x_2');
    title(['x_2 vs Iterations, ' label]);
    
    
    figure('Name', ['Contour & Gradient Descent Path - ' label]);
    hold on;
    x1_grid = linspace(-3, 3, 200);
    x2_grid = linspace(-3, 3, 200);
    [X1_grid, X2_grid] = meshgrid(x1_grid, x2_grid);
    Z_grid = f0(X1_grid, X2_grid);
    contour(X1_grid, X2_grid, Z_grid, 30);
    colorbar;
    plot(x1_seq, x2_seq, 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r', 'MarkerSize', 4);
    xlabel('x_1'); ylabel('x_2');
    title(['Contour of f with Gradient Descent Path, ' label]);
    hold off;
end


function y = f0(x1, x2)
    y = x1.^2 + 3 .* x1 .* x2 + 9 .* x2.^2 + 2 .* x1 - 5 .* x2;
end


function [x_hist, f_hist] = run_gradient_descent(x0, gamma, maxIter)
    
    x = x0;
    x_hist = zeros(2, maxIter+1);
    f_hist = zeros(1, maxIter+1);
    x_hist(:,1) = x;
    f_hist(1) = f0(x(1), x(2));
    
    for k = 1:maxIter
        g = [2*x(1) + 3*x(2) + 2; 3*x(1) + 18*x(2) - 5];
        x = x - gamma * g;
        x_hist(:, k+1) = x;
        f_hist(k+1) = f0(x(1), x(2));
    end
end
%Part 1e check for part 1b
syms x1 x2
eq1 = 2*x1 + 3*x2 + 2 == 0;
eq2 = 3*x1 + 18*x2 - 5 == 0;
sol = solve([eq1, eq2], [x1, x2]);

x_star = double([sol.x1; sol.x2]);
x1_min = double(sol.x1);
x2_min = double(sol.x2);

x_star = [x1_min; x2_min];
grad_f_star = [2*x_star(1) + 3*x_star(2) + 2;
               3*x_star(1) + 18*x_star(2) - 5];

fprintf('\n=== KKT (First-Order) Optimality Check ===\n');
disp(['Gradient at minimizer: [', num2str(grad_f_star(1)), ', ', num2str(grad_f_star(2)), ']']);

if norm(grad_f_star) < 1e-6
    disp('Gradient is zero at minimizer — First-order optimality satisfied.');
else
    disp('Gradient is not zero — Check for errors in calculation.');
end


 

