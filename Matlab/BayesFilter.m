function [map, posterior] = BayesFilter(emg, prior, pbins, param)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Improved version of Bayes-Chapman-Kolmogorov filter proposed in      %
% T. Sanger 2007                                                       % 
% Author: David Hofmann   -   david@nld.ds.mpg.de                      %
% Affiliation: Max Planck Institute for Dynamics and Self-Organization %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% emg:   N dimensional data vector (can be one sample, N = 1).
%        N.B. if N gets too large, numerical problems are encountered when 
%        computing the likelihood!
%        It is suggested to have N no larger than 40.
% prior: histogram distribution. Must be assigned to initial guess (usually
%        flat distribution for first iteration). For all following
%        iterations prior is assigned to the posterior from iteration
%        before.
% pbins: prior bins, values corresponding to the bins of the distribution 
%        of sigma;
%        NOTE: we assume the signals normalized by division with sigmaMVC 
%        -> support of 'sigma' distribution is always between 0 and 1 and 
%        must be multiplied by sigmaMVC to get the actual amplitude estimate
% param: parameter structure containing
%   .alpha:    'diffusion' parameter.
%   .beta:     Master equation transition rate.
%   .sigmaMVC: sigma during maximum voluntary contraction.
%   .sf:       sampling frequency.
%   .model:    defines the likelihood function, either gauss or laplace.
%   .pointmax: 'true' point estimation via maximum, 'false' via expectation value
%   .bins:     number of bins of sigma distribution

    % compute time step depending on number of samples in emg and on
    % sampling rate.
    N = length(emg);
    dt = param.sf^-1*N;
    dsigma = pbins(2) - pbins(1);   % delta sigma
    % propagate prior according to time evolution equation
    newPrior = (dt * param.alpha * ...
        ([prior(1);prior(1:end-1)] - 2*prior + [prior(2:end);prior(end)])/ ...
        dsigma^2 + dt * param.beta + (1 - dt * param.beta) * prior );

    % compute likelihood (assumption of independence => product of
    % likelihoods for each emg sample)
    switch param.model
        case 'Gauss'
            likelihood = exp(-0.5 * sum(emg.^2) ./ pbins.^2) ./ pbins.^N;
        case 'Laplace'
            likelihood = exp(-sum(abs(emg))./pbins) ./ pbins.^N;
        otherwise
            error('Unknown option for likelihood model');
    end
    if pbins(1) == 0
        likelihood(1) = 0;   % reset probability to 0 in case of divergence due to sigma = 0
    end

    posterior = likelihood .* newPrior;
    posterior = posterior/norm(posterior,1);  % normalize distribution
    % point estimation via maximum or expectation value
    if param.pointmax
        [~, idx] = max(posterior);
        map = pbins(idx);
    else
        map = sum(posterior.*pbins);
    end
end
