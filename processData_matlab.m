clear
clc

%% Results: Cost Esimate
filename = 'result_Cost_1.mat';
load(filename);

figure(1)
semilogy(val_mean_squared_error)
hold on;
semilogy(mean_squared_error)

legend({'Test Set','Training Set'})
xlabel('Epoch')
ylabel('Mean Squared Error');
grid on

%% Results: Discrete Variables Estimate
filename = 'result_Discrete_1.mat';
load(filename);

figure(2)
plot(100*val_acc);
hold on;
plot(100*acc);

legend({'Test Set','Training Set'})
xlabel('Epoch')
ylabel('Percent Accuracy (%)');
