both_feasible0 = load('both_feasible0.txt');
figure;
hist(both_feasible0, 100);
xlabel('time (s)', 'FontSize', 14);
ylabel('count', 'FontSize', 14);
set(gca, 'FontSize', 14);

both_infeasible0 = load('both_infeasible0.txt');
figure;
[~, edges] = histcounts(log10(both_infeasible0), 100);
histogram(both_infeasible0, 10.^edges);
xlabel('time (s)', 'FontSize', 14);
ylabel('count', 'FontSize', 14);
set(gca, 'FontSize', 14);
set(gca, 'xscale', 'log');

figure;
hist(both_infeasible0(both_infeasible0 < 0.1), 100);
xlabel('time (s)', 'FontSize', 14);
ylabel('count', 'FontSize', 14);
set(gca, 'FontSize', 14);

figure;
ee_position_error0 = load('ee_position_error0.txt');
hist(ee_position_error0 * 100, 100);
xlabel('position error (cm)', 'FontSize', 14);
ylabel('count', 'FontSize', 14);
set(gca, 'FontSize', 14);

figure;
ee_orientation_error0 = load('ee_orientation_error0.txt');
hist(ee_orientation_error0 / pi * 180, 100);
xlabel('orientation error (deg)', 'FontSize', 14);
ylabel('count', 'FontSize', 14);
set(gca, 'FontSize', 14);