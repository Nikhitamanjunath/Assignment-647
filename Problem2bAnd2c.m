clear; clc; close all;
run('topology.m');
A = zeros(Num_Links, Num_Flows);
for i = 1:Num_Flows
    for link = Flow_Path(i, :)
        if link > 0
            A(link, i) = 1;
        end
    end
end
lambda = zeros(Num_Links, 1);
alpha = 0.01;
max_iters = 4000;
flow_rates_history = zeros(max_iters, Num_Flows);
lambda_history = zeros(max_iters, Num_Links);

for it = 1:max_iters
    
    x = zeros(Num_Flows, 1);
    for i = 1:Num_Flows
        links = find(A(:, i));
        total_lambda = sum(lambda(links));
        if total_lambda > 0
            x(i) = Flow_Weight(i) / total_lambda;
        else
            x(i) = 10;
        end
    end
    flow_rates_history(it, :) = x';
    lambda_history(it, :) = lambda';
    
    lambda = max(lambda + alpha * (A * x - Link_Capacity'), 0);
end
final_flows = flow_rates_history(end, :)'

figure;
plot(flow_rates_history, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Flow Rate');
title('Flow Rate Convergence');
legend(arrayfun(@(i) sprintf('Flow %d', i), 1:Num_Flows, 'UniformOutput', false));
grid on;

figure;
plot(lambda_history, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Lambda Value');
title('Dual Variable (Lambda) Convergence');
legend(arrayfun(@(i) sprintf('\\lambda_{%d}', i), 1:Num_Links, 'UniformOutput', false));
grid on;

Ax = A * final_flows;
primal_residual = Ax - Link_Capacity(:);
fprintf('Value for Maximum constraint violation (Ax <= c): %.2e\n', max(primal_residual));

fprintf('Value for Minimum dual variable: %.2e\n', min(lambda));

comp_slackness = lambda' * (Ax - Link_Capacity(:));
fprintf('Value for Complementary slackness : %.2e\n', comp_slackness);

w = Flow_Weight(:);
stationarity_error = zeros(Num_Flows, 1);
for i = 1:Num_Flows
    links = find(A(:, i));
    total_lambda = sum(lambda(links));
    if total_lambda > 0
        x_check = w(i) / total_lambda;
    else
        x_check = 10;
    end
    stationarity_error(i) = abs(final_flows(i) - x_check);
end
fprintf('Value for Maximum stationarity error: %.2e\n', max(stationarity_error));
if max(primal_residual) < 1e-3 && ...
   min(lambda) > -1e-6 && ...
   abs(comp_slackness) < 1e-3 && ...
   max(stationarity_error) < 1e-6
   disp('The solution is optimal, KKT conditions satisfied');
else
   disp('The solution does not satisfy KKT conditions');
end


