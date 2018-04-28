load('data.mat')
% -------------  Parameters for Bayesian filter  -------------
param.sf = samplingfreq;  % set sampling frequency
param.bins = 100;         % output i.e. parameter quantization levels
param.alpha = 10^-10;     % sets diffusion rate
param.beta = 10^-40;      % sets probability of sudden jumps
param.model = 'Laplace';  % choose between 'Laplace' and 'Gauss'
param.pointmax = false;   % false: use expectation value as point estimation, 
                          % true: use maximum of posterior as point estimation
param.sigmaMVC = 0.5;     % maximum voluntary contraction amplitude value
% -------------  Parameters for Bayesian filter END  -------------

pri = ones(param.bins,1)/param.bins;         % define uniform prior
x = x/param.sigmaMVC;   % rescale data with respect to sigma MVC. This 
% helps avoiding numerical problems in case of the raw data being 
% especially if not single values are processed with the function 
% BayesFilter but instead multiple EMG measurements at a time. Refer also 
% to the comments in BayesFilter.m.
sig = linspace(param.bins^-1,1,param.bins)';  % sigma (amplitude) axis can 
% start at 0 or param.bins^-1. This is a matter of taste. Note howeverthat 
% sigma=0 will always have zero probability by definition (0 is an absorbing 
% boundary) due to the divergence at 0 when computing the likelihood. Refer
% also to the comments in BayesFilter.m.
bayesSTD = zeros(size(x));
% perform the filtering
for j = 1:size(x,2)
    for i = 1:size(x,1)
        [bayesSTD(i,j), pri] = BayesFilter(x(i,j), pri, sig, param);
    end
end
bayesSTD = bayesSTD * param.sigmaMVC;
% plot results
time = [0:length(x)-1]/param.sf;
plot(time, bayesSTD,'.','linewidth',1)
axis tight
ylim([0 param.sigmaMVC])
xlabel('\fontsize{14}time [s]')
ylabel('\fontsize{14}\sigma [mV]')
title(['\fontsize{12}Signal filtered with ' param.model ', \alpha = ' ...
    num2str(param.alpha) ', \beta = ' num2str(param.beta)])
