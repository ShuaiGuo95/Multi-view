db_path = 'C:\Users\15617\Desktop\MvLLS_datasets';
% db_path = '.';

num_view = 3;
X_tr = cell(1, num_view);
X_te = cell(1, num_view);
Y_tr = cell(1, num_view);
Y_te = cell(1, num_view);
trainset_path = cell(1, num_view);

% % sensor
% trainset_path{1} = [db_path, '\FeatureRGB.mat'];
% trainset_path{2} = [db_path, '\FeatureD.mat'];
% testset_path{1} = [db_path, '\FeatureRGB.mat'];
% testset_path{2} = [db_path, '\FeatureD.mat'];

% features
trainset_path{1} = [db_path, '\MobileNet_st.mat'];
trainset_path{2} = [db_path, '\InceptionV3_st.mat'];
trainset_path{3} = [db_path, '\Xception_st.mat'];
testset_path{1} = [db_path, '\MobileNet_st.mat'];
testset_path{2} = [db_path, '\InceptionV3_st.mat'];
testset_path{3} = [db_path, '\Xception_st.mat'];

% % poses
% trainset_path{1} = [db_path, '\MobileNet_st.mat'];
% trainset_path{2} = [db_path, '\MobileNet_hr.mat'];
% trainset_path{3} = [db_path, '\MobileNet_hl.mat'];
% testset_path{1} = [db_path, '\MobileNet_st.mat'];
% testset_path{2} = [db_path, '\MobileNet_hr.mat'];
% testset_path{3} = [db_path, '\MobileNet_hl.mat'];

flag = 600;
for i = 1:num_view
    temp = cell2mat(struct2cell(load(trainset_path{i})));
    X_tr{i} = temp(1:flag, 1:size(temp, 2)-1);
    Y_tr{i} = temp(1:flag, size(temp, 2));
    X_te{i} = temp(flag+1:size(temp, 1), 1:size(temp, 2)-1);
    Y_te{i} = temp(flag+1:size(temp, 1), size(temp, 2));
end

dim = 100;
numofneighbor = min(flag, 200);
Graph_poses = le_jaccard_graph(X_tr, Y_tr, numofneighbor, 300);
% save('Graph_poses.mat', 'Graph_poses');
% [TrainingAccuracyp1, TestingAccuracyp1, plist1] = ELM( [Y_tr{1}, X_tr{1}], [Y_te{1}, X_te{1}], 1, 8000, 'sig');

%% MvLE
% load('Graph_poses.mat');
% dist = [];
% for i = 1:num_view
%     dist(i) = size(X_tr{i}, 1);
% end
% D = cell2mat(cellfun(@diag, cellfun(@sum, mat2cell(Graph_poses,dist,dist), 'UniformOutput',false), 'UniformOutput',false));

D = diag(sum(Graph_poses));

L = D - Graph_poses;
L = max(L ,L');
[eigvector, eigvalue] = eig(L, D);%计算特征向量
[eigvalue, ansind] = sort(diag(eigvalue));
mappedX = eigvector(:,ansind(2:min(dim, flag)+1));

%% MvLLS
% load('Graph_poses.mat');
% mappedX = PLS_Bases_3(Graph_poses, dim, dist);

%% test
[Xt1] = lle_outofsample(X_tr{1}, mappedX(1:flag, :), X_te{1}, Graph_poses(1:flag, 1:flag), min(numofneighbor, flag)-1, 'sig'); 
[TrainingAccuracyp3, TestingAccuracyp3, plist3] = ELM( [Y_tr{1}, mappedX(1:flag, :)], [Y_te{1}, Xt1], 1, 8000, 'sig');
% % SVM
% model = svmtrain(Y_tr{1}, mappedX(1:flag, :));
% [~, acc_SVM, ~] = svmpredict(Y_te{1}, Xt1, model);