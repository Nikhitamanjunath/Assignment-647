% Part C
gamma_c = 0.001;      % step size -  gamma =  0.001, 0.01, 0.5
max_iter = 500;
[x1_grid, x2_grid] = meshgrid(-5:0.1:5, -5:0.1:5);
f_vals = f0(x1_grid, x2_grid);
grad_f = @(x1, x2) df0(x1, x2);
x = [1; 1];         % Starting point
traj = zeros(2, max_iter);
for k = 1:max_iter
    grad = grad_f(x(1), x(2));     
    grad = grad(:);                
    x_new = x - gamma_c * grad;     % Gradient step
    x = projection(x_new);         
    traj(:,k) = x;                 
end

figure('Name', sprintf('Gradient Projection for gamma = %.3f', gamma_c));
subplot(2,1,1);
plot(1:max_iter, traj(1,:), 'r', 1:max_iter, traj(2,:), 'b');
legend('x1(k)', 'x2(k)');
xlabel('Iteration'); ylabel('Value');
title(sprintf('Evolution of x1 and x2 (Step size %.3f)', gamma_c));

subplot(2,1,2);
contour(x1_grid, x2_grid, f_vals, 30); hold on;

plot(traj(1,:), traj(2,:), 'k-o', 'LineWidth', 1.2, 'MarkerSize', 3);
xlabel('x1'); ylabel('x2');
title('Contour and Gradient Projection Trajectory');
grid on; axis([0 3 0 3]);

%% Part E- Verification for Part C
x_proj = traj(:,end);
g1 = 2*x_proj(1) + x_proj(2) - 3;
g2 = x_proj(1) + 2*x_proj(2) - 3;
grad_proj = grad_f(x_proj(1), x_proj(2));
is_feasible = g1 >= -1e-3 && g2 >= -1e-3 && x_proj(1) >= -1e-3 && x_proj(2) >= -1e-3;
projected_grad_norm = norm(grad_proj);
fprintf('\nVerifying Part C \n');
fprintf('\nProjected gradient norm: %.2e \n', projected_grad_norm);
if is_feasible
    fprintf('It is a feasible solution \n');
else
    fprintf('It is not feasible solution \n');
end


