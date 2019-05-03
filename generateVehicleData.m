%% Description:
%  Generates batch of solutions to mixed-integer quadratic program
%  representing the control of a hybrid powertrain.
%  Parameter varied (p) is the power demand

clear
clc
close all

rng('default')
rng(1)
vehicle_data

%Various indices
l1 = 72;
l4 = 503;

%Replace indices 73-144 of vector b with a parameter that we will update every time.
%Parameter represents the power demand over next 72 steps
p = sdpvar(72,1);
bp = [b(1:72); p; b((72*2+1):575)];
%% Parametric mixed-integer quadratic program
z = binvar(l1,1);
x = sdpvar(n-l1,1);
x_cvx = [z;x];

%Constraints
con2 = x_cvx((l1+1) : (l1+l4)) >= 0;
con3 = A*x_cvx == bp;
%Objective
obj = (0.5 * x_cvx' * P * x_cvx + q' * x_cvx + r);
%Solver settings
options = sdpsettings('solver','gurobi','verbose',0);
options.gurobi.TimeLimit = 3;
options.gurobi.MIPGap = 1e-2;
%Parametric optimizer
mip = optimizer([con2;con3], obj, options, {p}, {obj, x_cvx});


%% Batch run of solutions
solutionCountTotal = 0;
solutionCountFile = 0;
solutionPerFile = 10e3;
solutionSaveRate = 100;
solutionSetNumber = 1;

dataToSave = ['PowerCommand, OptimalCost, SolveTime, Control'];

iterCount = 0;
while solutionCountTotal <= 1e7
    iterCount = iterCount + 1;
    filename = ['vehicleCostToGo_Batch1_' num2str(solutionSetNumber)];
    
    if mod(iterCount, 10) == 0;
        disp(['# iterations: ' num2str(iterCount) ' # solutions: ' num2str(solutionCountTotal)]);
    end
    if mod(solutionCountFile, solutionSaveRate) == 0 && solutionCountFile > 0
        save(filename, 'PowerCommand', 'OptimalCost', 'SolveTime', 'Control');
    end
    
    if mod(solutionCountTotal,solutionPerFile) == 0 && solutionCountFile > 0
        save(filename, 'PowerCommand', 'OptimalCost', 'SolveTime', 'Control');
        %Move to next file set number
        solutionSetNumber = solutionSetNumber + 1;
        solutionCountFile = 0;
        
        %Initialize data
        PowerCommand = [];
        Control = [];
        OptimalCost = [];
        SolveTime = [];
    end
%% Generate a new random power profile and solve
n_int = 6;
Pmax = 1.8;
npower = Pmax*2*(-0.5+rand(n_int+1,1));
b0 = [];
for k = 1:(n_int)
    nl = linspace(npower(k), npower(k+1),72/n_int+1);
    b0 = [b0 nl(1:(end-1))];
end

b0 = b0(1:72);

bp = b;
bp(73:(73+71)) = b0;

tic
[output errorcode] = mip(b0(:));
tt = toc;
%% If solved to within tolerance, store results
obj = output{1};
Control = output{2};
if errorcode == 0 
    solutionCountTotal = solutionCountTotal+1;
    solutionCountFile = solutionCountFile+1;
    
    PowerCommand(:,solutionCountFile) = npower;
    Control(:,solutionCountFile) = output{2};
    OptimalCost(:,solutionCountFile) = obj;
    SolveTime(:,solutionCountFile) = tt;
end

end
