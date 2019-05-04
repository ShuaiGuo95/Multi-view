% %%%%%%%%%%%%%%%%%% Generating data
% temp = load('FeatureRGB_tr.mat');
% RGBfeature_tr = temp.FeatureRGB_tr(:, 1:4096);
% RGBfeature_tr(:, all(RGBfeature_tr==0,1)) = [0.1];
% RGBlabel_tr = temp.FeatureRGB_tr(:, 4097);
% 
% temp = load('FeatureRGB_te.mat');
% RGBfeature_te = temp.FeatureRGB_te(:, 1:4096);
% RGBfeature_te(:, all(RGBfeature_te==0,1)) = [0.1];
% RGBlabel_te = temp.FeatureRGB_te(:, 4097);
% 
% temp = load('FeatureD_tr.mat');
% Dfeature_tr = temp.FeatureD_tr(:, 1:4096);
% Dfeature_tr(:, all(Dfeature_tr==0,1)) = [0.1];
% Dlabel_tr = temp.FeatureD_tr(:, 4097);
% 
% temp = load('FeatureD_te.mat');
% Dfeature_te = temp.FeatureD_te(:, 1:4096);
% Dfeature_te(:, all(Dfeature_te==0,1)) = [0.1];
% Dlabel_te = temp.FeatureD_te(:, 4097);

%%%%%%%%%%%%%%%%%% 初始化图Graph
% X = {RGBfeature_tr, Dfeature_tr};
% Y = {RGBlabel_tr, Dlabel_tr};
dim = 1000;
numofneighbor = 150;
Graph = le_jaccard_graph(X, Y, numofneighbor, 1000);
% save('Graph.mat', 'Graph');
% 
% %%%%%%%%%%%%%%%%%%% PRE
% RGBfeature_tr = RGBfeature_tr';
% RGBfeature_te = RGBfeature_te';
% Dfeature_tr = Dfeature_tr';
% Dfeature_te = Dfeature_te';
% mappedX = PLS_Bases_3(Graph, RGBfeature_tr, Dfeature_tr, dim, 0, 0, 'no_kernel');
% 
% RGBfeature_tr = RGBfeature_tr';
% RGBfeature_te = RGBfeature_te';.
% 
% Dfeature_tr = Dfeature_tr';
% Dfeature_te = Dfeature_te';

% load('mappedX_nokernel.mat');
% load('Graph.mat');
D = sum(Graph);
D = diag(D);
L = D - Graph;
L=max(L ,L');
[eigvector, eigvalue] = eig(L, D);%计算特征向量
[eigvalue, ansind] = sort(diag(eigvalue));
mappedX = eigvector(:,ansind(2:dim+1));

% [TrainingAccuracyp1, TestingAccuracyp1, plist1] = ELM( [RGBlabel_tr(:, :), RGBfeature_tr(:, :)], [RGBlabel_te(:, :), RGBfeature_te(:, :)], 1, 800, 'sig');
% [TrainingAccuracyp2, TestingAccuracyp2, plist2] = ELM( [Dlabel_tr(:, :), Dfeature_tr(:, :)], [Dlabel_te(:, :), Dfeature_te(:, :)], 1, 800, 'sig');

% [Zt1, Xt1] = elmDR(RGBfeature_tr(:, :), mappedX(1:2970, :), RGBfeature_te(:, :), 1, 1200, 'agauss'); % 1000 sig
% [Zt2, Xt2] = elmDR(Dfeature_tr(:, :), mappedX(2971:5940, :), Dfeature_te(:, :), 1, 1200, 'agauss');

[Xt1] = lle_outofsample(RGBfeature_tr(:, :), mappedX(1:2970, :), RGBfeature_te(:, :), Graph(1:2970, 1:2970), numofneighbor, 'sig'); 
% [Xt2] = lle_outofsample(Dfeature_tr(:, :), mappedX(2971:5940, :), Dfeature_te(:, :), Graph(2971:5940, 2971:5940), 200, 'sig');

[TrainingAccuracyp3, TestingAccuracyp3, plist3] = ELM( [RGBlabel_tr(:, :), mappedX(1:2970, :)], [RGBlabel_te(:, :), Xt1(:, :)], 1, 8000, 'sig');
% [TrainingAccuracyp4, TestingAccuracyp4, plist4] = ELM( [Dlabel_tr(:,:), mappedX(2971:5940, :)], [Dlabel_te(:, :), Xt2(:, :)], 1, 8000, 'sig');