function [K,dK] = gplite_covfun(hyp,X,covfun,Xstar)
%GPLITE_COVFUN Covariance function for lite Gaussian Process regression.
%   M = GPLITE_MEANFUN(HYP,X,MEANFUN) computes the GP mean function
%   MEANFUN evaluated at test points X. HYP is a single column vector of mean 
%   function hyperparameters. MEANFUN can be a scalar or a character array
%   specifying the mean function, as follows:
%
%      MEANFUN          MEAN FUNCTION TYPE                  HYPERPARAMETERS
%      0 or 'zero'      zero                                0
%      1 or 'const'     constant                            1
%      2 or 'linear'    linear                              D+1
%      3 or 'quad'      quadratic                           2*D+1
%      4 or 'negquad'   negative quadratic, centered        2*D+1
%      5 or 'posquad'   positive quadratic, centered        2*D+1
%      6 or 'se'        squared exponential                 2*D+2
%      7 or 'negse'     negative squared exponential        2*D+2
%      function_handle  custom                              NMEAN
%
%   MEANFUN can be a function handle to a custom mean function.
%
%   [M,DM] = GPLITE_MEANFUN(HYP,X,MEANFUN) also computes the gradient DM 
%   with respect to GP hyperparamters. DM is a N-by-NMEAN matrix, where
%   each row represent the gradient with respect to the NMEAN hyperparameters
%   for each one of the N test point.
%
%   NMEAN = GPLITE_MEANFUN([],X,MEANFUN) returns the number of mean function
%   hyperparameters requested by mean function MEANFUN.
%
%   [NMEAN,MEANINFO] = GPLITE_MEANFUN([],X,MEANFUN,Y), where X is the matrix
%   of training inputs and Y the matrix of training targets, also returns a 
%   struct MEANINFO with additional information about mean function
%   hyperparameters, with fields
% 
%      LB  Hyperparameter lower bounds
%      UB  Hyperparameter upper bounds

if nargin < 4; Xstar = []; end

if isa(covfun,'function_handle')
    if nargout > 1
        [K,dK] = covfun(hyp,X);
    else
        K = covfun(hyp,X);
    end
    return;
end

[N,D] = size(X);            % Number of training points and dimension

% Read number of mean function hyperparameters
switch covfun(1)
    case {0,'0','zero'}
        Ncov = D+1;
        covfun = 0;
    case {1,'1','const'}
        Ncov = 1;
        meanfun = 1;
%    otherwise
%        if isnumeric(meanfun); meanfun = num2str(meanfun); end
%        error('gplite_meanfun:UnknownMeanFun',...
%            ['Unknown mean function identifier: [' meanfun '].']);
end

% Return number of mean function hyperparameters and additional info
if ischar(hyp)
    K = Ncov;
    if nargout > 1
        ToL = 1e-6;
        Big = exp(3);
        dm.LB = -Inf(1,Ncov);
        dm.UB = Inf(1,Ncov);
        dm.PLB = -Inf(1,Ncov);
        dm.PUB = Inf(1,Ncov);
        dm.x0 = NaN(1,Ncov);
        
        if meanfun >= 1                     % m0
            h = max(y) - min(y);    % Height
            dm.LB(1) = min(y) - 0.5*h;
            dm.UB(1) = max(y) + 0.5*h;
            dm.PLB(1) = quantile1(y,0.1);
            dm.PUB(1) = quantile1(y,0.9);
            dm.x0(1) = median(y);
        end
        
        if meanfun >= 2 && meanfun <= 3     % a for linear part
            w = max(X) - min(X);    % Width
            delta = w./h;
            dm.LB(2:D+1) = -delta*Big;
            dm.UB(2:D+1) = delta*Big;
            dm.PLB(2:D+1) = -delta;
            dm.PUB(2:D+1) = delta;
            if meanfun == 3
                dm.LB(D+2:2*D+1) = -(delta*Big).^2;
                dm.UB(D+2:2*D+1) = (delta*Big).^2;                
                dm.PLB(D+2:2*D+1) = -delta.^2;
                dm.PUB(D+2:2*D+1) = delta.^2;                
            end
            
            
            w = max(X) - min(X);                    % Width

            dm.LB(2:D+1) = min(X) - 0.5*w;          % xm
            dm.UB(2:D+1) = max(X) + 0.5*w;
            dm.PLB(2:D+1) = min(X);
            dm.PUB(2:D+1) = max(X);
            dm.x0(2:D+1) = median(X);

            dm.LB(D+2:2*D+1) = log(w) + log(ToL);   % omega
            dm.UB(D+2:2*D+1) = log(w) + log(Big);
            dm.PLB(D+2:2*D+1) = log(w) + 0.5*log(ToL);
            dm.PUB(D+2:2*D+1) = log(w);
            dm.x0(D+2:2*D+1) = log(std(X));
            
            if meanfun == 6 || meanfun == 7                
                dm.LB(2*D+2) = log(h) + log(ToL);   % h
                dm.UB(2*D+2) = log(h) + log(Big);
                dm.PLB(2*D+2) = log(h) + 0.5*log(ToL);
                dm.PUB(2*D+2) = log(h);
                dm.x0(2*D+2) = log(std(y));   
            end
            
            if meanfun == 8 || meanfun == 9                
                dm.LB(2*D+1+(1:D)) = min(X) - 0.5*w;          % xm_se
                dm.UB(2*D+1+(1:D)) = max(X) + 0.5*w;
                dm.PLB(2*D+1+(1:D)) = min(X);
                dm.PUB(2*D+1+(1:D)) = max(X);
                [~,idx_max] = max(y);
                dm.x0(2*D+1+(1:D)) = X(idx_max,:);

                dm.LB(3*D+1+(1:D)) = log(w) + log(ToL);   % omega_se
                dm.UB(3*D+1+(1:D)) = log(w) + log(Big);
                dm.PLB(3*D+1+(1:D)) = log(w) + 0.5*log(ToL);
                dm.PUB(3*D+1+(1:D)) = log(w);
                dm.x0(3*D+1+(1:D)) = log(std(X));
                
                dm.LB(4*D+2) = -Big*h;   % h_se
                dm.UB(4*D+2) = Big*h;
                dm.PLB(4*D+2) = -h;
                dm.PUB(4*D+2) = h;
                dm.x0(4*D+2) = min(std(y),h);   
            end
            
        end
        
        % Plausible starting point
        idx_nan = isnan(dm.x0);
        dm.x0(idx_nan) = 0.5*(dm.PLB(idx_nan) + dm.PUB(idx_nan));
        
        dm.meanfun = meanfun;
        switch meanfun
            case 0
                dm.meanfun_name = 'zero';
            case 1
                dm.meanfun_name = 'const';
            case 2
                dm.meanfun_name = 'linear';
            case 3
                dm.meanfun_name = 'quad';
            case 4
                dm.meanfun_name = 'negquad';
            case 5
                dm.meanfun_name = 'posquad';
            case 6
                dm.meanfun_name = 'se';
            case 7
                dm.meanfun_name = 'negse';
            case 8
                dm.meanfun_name = 'negquadse';
            case 9
                dm.meanfun_name = 'posquadse';
        end
        
    end
    
    
    
    return;
end

[Nhyp,Ns] = size(hyp);      % Hyperparameters and samples

if Nhyp ~= Ncov
    error('gplite_meanfun:WrongMeanHyp', ...
        ['Expected ' num2str(Ncov) ' mean function hyperparameters, ' num2str(Nhyp) ' passed instead.']);
end
if Ns > 1
    error('gplite_meanfun:nosampling', ...
        'Mean function output is available only for one-sample hyperparameter inputs.');
end

% Compute mean function gradient wrt hyperparameters only if requested
compute_grad = nargout > 1;

if compute_grad     % Allocate space for gradient
    dK = zeros(N,N,Ncov);    
end

% Compute mean function
ell = exp(hyp(1:D));
sf2 = exp(2*hyp(D+1));

switch meanfun
    case 0  % SE ard
        if isempty(Xstar)        
            K = sq_dist(diag(1./ell)*X');
        elseif ischar(Xstar)
            K = zeros(size(X,1),1);
        else
            K = sq_dist(diag(1./ell)*X',diag(1./ell)*Xstar');
        end
        K = sf2 * exp(-K/2);
            
        if compute_grad
            for i = 1:D             % Grad of cov length scales
                dK(:,:,i) = K .* sq_dist(X(:,i)'/ell(i));
            end
            dK(:,:,i) = 2*K;        % Grad of cov output scale
        end        
end

end