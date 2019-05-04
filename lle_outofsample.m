function [Y_te] = lle_outofsample(X_tr, Y_tr, X_te, tempG, K, kernel)

switch lower(kernel)
    case {'no_kernel'}
        X_tr = X_tr;
        X_te = X_te;
    case {'sig','sigmoid'}
        X_tr = 1 ./ (1 + exp(-X_tr));
        X_te = 1 ./ (1 + exp(-X_te));
    case {'sin','sine'}
        X_tr = sin(X_tr);
        X_te = sin(X_te);
    case {'hardlim'}
        X_tr = double(hardlim(X_tr));
        X_te = double(hardlim(X_te));
    case {'tribas'}
        X_tr = tribas(X_tr);
        X_te = tribas(X_te);
    case {'radbas'}
        X_tr = radbas(X_tr);
        X_te = radbas(X_te);
end

tempG = exp(-tempG);
tempG = sum(tempG, 1)/200;
tempG = repmat(tempG, size(X_te, 1), 1);
[~, ind] = sort(pdist2(X_te, X_tr, 'cityblock') + tempG, 2);
neighborhood = ind(:, 2:(1+K));
neighborhood = neighborhood';

X_tr = X_tr';
N = size(X_te, 1);
X_te = X_te';
W = zeros(K,N);
tol = 0;

for ii=1:N
   z = X_tr(:,neighborhood(:,ii))-repmat(X_te(:,ii),1,K); % shift ith pt to origin
   C = z'*z;                                        % local covariance
   C = C + eye(K,K)*tol*trace(C);                   % regularlization (K>D)
   W(:,ii) = pinv(C)*ones(K,1);                           % solve Cw=1
   W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
end;

W = W';
% [~, Wfinal] = elmDR(Wtr, Wte, W, 1, 800, 'sig'); % 1000 sig
% Wfinal = Wfinal./repmat(sum(Wfinal, 2), 1, size(Wfinal, 2));
% save('Wfinal.mat', 'Wfinal');
% save('W.mat', 'W');
Y_te = zeros(N, size(Y_tr, 2));
for i=1:N
    for j=1:K
        Y_te(i,:)=Y_te(i, :)+W(i, j)*Y_tr(neighborhood(j, i), :);
    end
end