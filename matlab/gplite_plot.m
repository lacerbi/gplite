function gplite_plot(gp)
%GPLITE_PLOT Visualize GP. (For now only 1D.)

[N,D] = size(gp.X);            % Number of training points and dimension
Ns = numel(gp.post);           % Hyperparameter samples
Nx = 1e3;                      % Number of points for visualization

% Loop over hyperparameter samples
ell = zeros(D,Ns);
for s = 1:Ns
    hyp = gp.post(s).hyp;
    ell(:,s) = exp(hyp(1:D));       % Extract length scales from HYP
end
ellbar = sqrt(mean(ell.^2,2));      % Mean length scale

lb = min(gp.X) - ellbar';
ub = max(gp.X) + ellbar';


xx = linspace(lb,ub,Nx)';
[~,~,fmu,fs2] = gplite_pred(gp,xx);

plot(xx,fmu,'-k','LineWidth',1); hold on;
plot(xx,fmu+1.96*sqrt(fs2),'-k','LineWidth',0.5); hold on;
plot(xx,fmu-1.96*sqrt(fs2),'-k','LineWidth',0.5); hold on;
scatter(gp.X,gp.y,'ok');

end