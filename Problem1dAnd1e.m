% Part D
x = [2; 2]; % Starting point [x1, x2]
grad_f = @(x1, x2) df0(x1,x2);
step_sizes = [0.01, 0.1, 0.6];  
labels = {'Too Small(\gamma=0.01)', 'Just Right(\gamma=0.1)', 'Too Large(\gamma=0.6)'};
max_iters = 100;
x_traj_all = cell(1, 3);
lambda_traj_all = cell(1, 3);
for idx = 1:3
    gamma = step_sizes(idx);
    lambda = [0; 0];
    lambda_traj = lambda';
    x_traj = [];
    for k = 1:max_iters
        L = @(x) f0(x(1), x(2)) + lambda(1)*(3 - 2*x(1) - x(2)) + lambda(2)*(3 - x(1) - 2*x(2));
        x = fmincon(L, [1; 1], [], [], [], [], [0; 0], []);
        
        x_traj = [x_traj; x'];
        
        c1 = 3 - 2*x(1) - x(2);
        c2 = 3 - x(1) - 2*x(2);
        grad_lambda = [c1; c2];
        lambda = max(0, lambda + gamma * grad_lambda);  
        lambda_traj = [lambda_traj; lambda'];
    end
    x_traj_all{idx} = x_traj;
    lambda_traj_all{idx} = lambda_traj;
end

figure;
for idx = 1:3
    subplot(3,1,idx);
    plot(0:max_iters-1, x_traj_all{idx}(:,1), 'b', 'LineWidth', 1.5); hold on;
    plot(0:max_iters-1, x_traj_all{idx}(:,2), 'g', 'LineWidth', 1.5);
    legend('x_1(k)', 'x_2(k)');
    xlabel('Iteration'); ylabel('Primal Variables');
    title(['Primal Evolution - ', labels{idx}]);
    grid on;
end
figure;
for idx = 1:3
    subplot(3,1,idx);
    plot(0:max_iters, lambda_traj_all{idx}(:,1), 'm', 'LineWidth', 1.5); hold on;
    plot(0:max_iters, lambda_traj_all{idx}(:,2), 'c', 'LineWidth', 1.5);
    legend('\lambda_1(k)', '\lambda_2(k)');
    xlabel('Iteration'); ylabel('Dual Variables');
    title(['Dual Evolution - ', labels{idx}]);
    grid on;
end

[x1_grid, x2_grid] = meshgrid(0:0.1:3, 0:0.1:3);
f_vals = f0(x1_grid, x2_grid);
figure;
for idx = 1:3
    subplot(1,3,idx);
    contour(x1_grid, x2_grid, f_vals, 50); hold on;
    traj = x_traj_all{idx};
    plot(traj(:,1), traj(:,2), 'r-o', 'LineWidth', 1.5);
    title(['Contour,Trajectory:', labels{idx}]);
    xlabel('x_1'); ylabel('x_2');
    axis([0 3 0 3]); axis equal; grid on;
end
%% Part E- Verification for Part D
clc; clear;
grad_f = @(x) [2*x(1) + 3*x(2) + 2; 3*x(1) + 18*x(2) - 5];
grad_c1 = [-2; -1];  
grad_c2 = [-1; -2];  

x_star = [1; 1];          
lambda_star = [1; 1];     

c1_val = 3 - 2*x_star(1) - x_star(2);
c2_val = 3 - x_star(1) - 2*x_star(2);

fprintf("1) Primal Feasibility:\n");
fprintf("   c1(x) = %.6f >= 0: %d\n", c1_val, c1_val >= 0);
fprintf("   c2(x) = %.6f >= 0: %d\n", c2_val, c2_val >= 0);

fprintf("\n2) Dual Feasibility:\n");
fprintf("   lambda_1 = %.6f >= 0: %d\n", lambda_star(1), lambda_star(1) >= 0);
fprintf("   lambda_2 = %.6f >= 0: %d\n", lambda_star(2), lambda_star(2) >= 0);

cs1 = lambda_star(1) * c1_val;
cs2 = lambda_star(2) * c2_val;
fprintf("\n3) Complementary Slackness:\n");
fprintf("   lambda_1 * c1(x) = %.6e ≈ 0: %d\n", cs1, abs(cs1) < 1e-6);
fprintf("   lambda_2 * c2(x) = %.6e ≈ 0: %d\n", cs2, abs(cs2) < 1e-6);

grad_L = grad_f(x_star) + lambda_star(1)*grad_c1 + lambda_star(2)*grad_c2;
fprintf("\n4) Stationarity:\n");
fprintf("   ∇L = [%.6e; %.6e] ≈ [0;0]: %d\n", ...
    grad_L(1), grad_L(2), norm(grad_L) > 1e-3);

fprintf("\n--- Final Result ---\n");
if c1_val >= 0 && c2_val >= 0 && ...
   all(lambda_star >= 0) && ...
   abs(cs1) < 1e-3 && abs(cs2) < 1e-3 && ...
   norm(grad_L) > 1e-3
    disp('KKT conditions satisfied — solution is optimal.');
else
    disp('KKT conditions NOT fully satisfied — solution may not be optimal.');
end

