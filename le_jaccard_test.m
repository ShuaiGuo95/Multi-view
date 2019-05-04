%用dense特征的30张和tcdw特征的30张做融合，用剩余40张做测试
TrainNum = 60; TestNum = 40; 
denseNum = 30; tcdwNum = 30;
%读入数据 取出每个特征的前3000幅，共30类，每类100张。
temp = load('corel10k-1K-dense.mat');
densefeature = temp.our_feature1(1:3000, :);
denselabel = reshape(repmat(1:30, [100, 1]), 3000, 1);

temp = load('corel10k-HSV.mat');
hsvfeature = temp.our_feature1(1:3000, :);
hsvlabel = denselabel;

temp = load('corel10k-TCDW.mat');
tcdwfeature = temp.our_feature1(1:3000, :);
tcdwlabel = denselabel;

densefeaturetr = []; densefeaturete = []; denselabeltr = []; denselabelte = [];
% hsvfeaturetr = []; hsvfeaturete = []; hsvlabeltr = []; hsvlabelte = [];
tcdwfeaturetr = []; tcdwfeaturete = []; tcdwlabeltr = []; tcdwlabelte = [];
% 数据分割
for i = 1:3000
    temp = mod(i-1, 100) +1;
    if temp > TrainNum
        densefeaturete = [densefeaturete; double(densefeature(i, :))]; denselabelte = [denselabelte; double(denselabel(i))];
        % hsvfeaturete = [hsvfeaturete; double(hsvfeature(i, :))]; hsvlabelte = [hsvlabelte; double(hsvlabel(i))];
        tcdwfeaturete = [tcdwfeaturete; double(tcdwfeature(i, :))]; tcdwlabelte = [tcdwlabelte; double(tcdwlabel(i))];
    end
    if temp <= denseNum
        densefeaturetr = [densefeaturetr; double(densefeature(i, :))]; denselabeltr = [denselabeltr; double(denselabel(i))];
        % hsvfeaturetr = [hsvfeaturetr; double(hsvfeature(i, :))]; hsvlabeltr = [hsvlabeltr; double(hsvlabel(i))];
    end
    if temp > 30 && temp <= 60
        tcdwfeaturetr = [tcdwfeaturetr; double(tcdwfeature(i, :))]; tcdwlabeltr = [tcdwlabeltr; double(tcdwlabel(i))];
    end
end

X = {densefeaturetr, tcdwfeaturetr};
Y = {denselabeltr, tcdwlabeltr};
size(X)
size(Y) 

% model11 = svmtrain(denselabeltr(1:2400, :), densefeaturetr(1:2400, :), '-c 1 -g 0.07 -b 1');
% [predict_label, accuracy11, dec_values] = svmpredict(denselabelte(1:600, :), densefeaturete(1:600, :), model11);
% 
% model12 = svmtrain(hsvlabeltr(1:2400,:), hsvfeaturetr(1:2400,:), '-c 1 -g 0.07 -b 1');
% [predict_label, accuracy12, dec_values] = svmpredict(hsvlabelte(1:600, :), hsvfeaturete(1:600, :), model12);
% 
% model13 = svmtrain(tcdwlabeltr(1:2400,:), tcdwfeaturetr(1:2400,:), '-c 1 -g 0.07 -b 1');
% [predict_label, accuracy13, dec_values] = svmpredict(tcdwlabelte(1:600, :), tcdwfeaturete(1:600, :), model13);

% [TrainingAccuracyp3, TestingAccuracyp3, plist3] = ELM( [denselabeltr(1:denseNum*30, :), densefeaturetr(1:denseNum*30, :)], [denselabelte(1:TestNum*30, :), densefeaturete(1:TestNum*30, :)], 1, 500, 'sig');
% [TrainingAccuracyp4, TestingAccuracyp4, plist4] = ELM( [tcdwlabeltr(1:tcdwNum*30, :), tcdwfeaturetr(1:tcdwNum*30, :)], [tcdwlabelte(1:TestNum*30, :), tcdwfeaturete(1:TestNum*30, :)], 1, 500, 'sig');
%LE_jaccard特征融合
mappedX = le_jaccard(X, Y, 15, 300);
% mappedX = load('eigvector.mat');
% mappedX = mappedX.eigvector;
mapped = mappedX;

% tr1 = []; tr2 = []; tr3 = [];
% te1 = []; te2 = []; te3 = [];
% for i = 1:3000
%     if mod(i-1, 100) +1 > 80
%         te1 = [te1; double(mapped(i, :))];
%     else
%         tr1 = [tr1; double(mapped(i, :))];
%     end
% end
% for i = 3001:6000
%     if mod(i-1, 100) +1 > 80
%         te2 = [te2; double(mapped(i, :))];
%     else
%         tr2 = [tr2; double(mapped(i, :))];
%     end
% end
% for i = 6001:9000
%     if mod(i, 100)>80 || mod(i, 100)==0
%         te3 = [te3; double(mapped(i, :))];
%     else
%         tr3 = [tr3; double(mapped(i, :))];
%     end
% end
%使用elmDR做测试集处理
[Zt1, Xt1] = elmDR(densefeaturetr(1:denseNum*30, :), mapped(1:denseNum*30, :), densefeaturete(1:TestNum*30, :), 1, 800, 'sig');
% [Zt2, Xt2] = elmDR(hsvfeaturetr(1:EachNum*30, :), mapped(EachNum*30+1:EachNum*60, :), hsvfeaturete(1:TestNum*30, :), 1, 800, 'sig');
[Zt2, Xt2] = elmDR(tcdwfeaturetr(1:tcdwNum*30, :), mapped(denseNum*30+1:denseNum*30+tcdwNum*30, :), tcdwfeaturete(1:TestNum*30, :), 1, 800, 'sig');

% mode211 = svmtrain(denselabeltr(1:2400, :), mapped(1:2400, :), '-c 1 -g 0.07 -b 1');
% [predict_label, accuracy21, dec_values] = svmpredict(denselabelte(1:600, :), Xt1(1:600, :), mode211);

% mode212 = svmtrain(hsvlabeltr(1:2400,:), mapped(2401:4800, :), '-c 1 -g 0.07 -b 1');
% [predict_label, accuracy22, dec_values] = svmpredict(hsvlabelte(1:600, :), Xt2(1:600, :), mode212);

% mode213 = svmtrain(tcdwlabeltr(1:2400,:), tr3(1:2400, :), '-c 1 -g 0.07 -b 1');
% [predict_label, accuracy23, dec_values] = svmpredict(tcdwlabelte(1:600, :), te3(1:600, :), mode213);

%融合之后的效果
[TrainingAccuracyp1, TestingAccuracyp1, plist1] = ELM( [denselabeltr(1:denseNum*30, :), mapped(1:denseNum*30, :)], [denselabelte(1:TestNum*30, :), Xt1(1:TestNum*30, :)], 1, 500, 'sig');
% [TrainingAccuracyp2, TestingAccuracyp2] = ELM( [hsvlabeltr(1:EachNum*30,:), mapped(EachNum*30+1:EachNum*60, :)], [hsvlabelte(1:TestNum*30, :), Xt2(1:TestNum*30, :)], 1, 500, 'sig');
[TrainingAccuracyp2, TestingAccuracyp2, plist2] = ELM( [tcdwlabeltr(1:tcdwNum*30,:), mapped(denseNum*30+1:denseNum*30+tcdwNum*30, :)], [tcdwlabelte(1:TestNum*30, :), Xt2(1:TestNum*30, :)], 1, 500, 'sig');
plist12 = [plist1, plist2];

%单特征的baseline 使用ELM分类器
[TrainingAccuracyp3, TestingAccuracyp3, plist3] = ELM( [denselabeltr(1:denseNum*30, :), densefeaturetr(1:denseNum*30, :)], [denselabelte(1:TestNum*30, :), densefeaturete(1:TestNum*30, :)], 1, 500, 'sig');
% [TrainingAccuracyp4, TestingAccuracyp4] = ELM( [hsvlabeltr(1:EachNum*30, :), hsvfeaturetr(1:EachNum*30, :)], [hsvlabelte(1:TestNum*30, :), hsvfeaturete(1:TestNum*30, :)], 1, 500, 'sig');
[TrainingAccuracyp4, TestingAccuracyp4, plist4] = ELM( [tcdwlabeltr(1:tcdwNum*30, :), tcdwfeaturetr(1:tcdwNum*30, :)], [tcdwlabelte(1:TestNum*30, :), tcdwfeaturete(1:TestNum*30, :)], 1, 500, 'sig');
plist34 = [plist3, plist4];
%一起训练，一起测试及分别测试
dhltr = [denselabeltr; tcdwlabeltr];
dhlte = [denselabelte; tcdwlabelte];
dhfte = [Xt1; Xt2];
[TrainingAccuracypzz, TestingAccuracypzz, plistzz] = ELM( [dhltr, mapped], [dhlte, dhfte], 1, 500, 'sig');
[TrainingAccuracypz1, TestingAccuracypz1, plistz1] = ELM( [dhltr, mapped], [denselabelte(1:TestNum*30, :), Xt1(1:TestNum*30, :)], 1, 500, 'sig');
[TrainingAccuracypz2, TestingAccuracypz2, plistz2] = ELM( [dhltr, mapped], [tcdwlabelte(1:TestNum*30, :), Xt2(1:TestNum*30, :)], 1, 500, 'sig');