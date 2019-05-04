function Graph = le_jaccard_graph(X, Y, n0, t)
    
    sumn = 0;
    %在每个特征内计算以下变量：
    allnear = [];%存储每个样本的近邻
    alllabel = [];%存储所有样本的label
    allnearlabel = [];%存储第i个样本所有近邻的标签
    allC = [];%allC(i)统计每个类型的近邻个数
    allmat = [];
    
    mindim = size(cell2mat(X(1)), 2);
    for i = 2:size(X, 2)
        mindim = min(mindim, size(cell2mat(X(i)), 2));
    end
    for i = 1:size(X, 2)
        i
        tempi = cell2mat(X(i));
        [tempi, ~] = apca(tempi, mindim);
        n(i) = size(tempi, 1);%每种特征的样本个数
        
        temps = pdist2(tempi, tempi);%计算二范数距离
        [~, ind] = sort(temps, 2);%计算近邻排序
        ind = ind(:,1:n0);%取前n0个样本作为近邻
        tempy = cell2mat(Y(i));
        
        tempnl = ind;
        tempallc = zeros(size(ind, 1), 30);
        for j = 1:size(tempnl, 1)
            for k = 1:size(tempnl, 2)
                tempnl(j, k) = tempy(tempnl(j, k));
                tempallc(j, tempnl(j, k)) = tempallc(j, tempnl(j, k))+1;
            end
        end
        
        allnear = [allnear; ind];
        alllabel = [alllabel; tempy];
        allnearlabel = [allnearlabel; tempnl];
        allC = [allC; tempallc];
        allmat = [allmat; tempi];
        sumn = sumn + n(i);%样本总数
    end
    
    S = zeros(sumn, sumn);
    for i = 1:sumn
        i
        tempi = allnearlabel(i, :);
        for j = i+1:sumn %i+1-->i
            tempj = allnearlabel(j, :);
            if ismember(alllabel(i), tempj) && ismember(alllabel(j), tempi) %i的label存在于j的近邻当中，且j的label存在于i的近邻当中
                S(i, j) = norm(allmat(i, :)-allmat(j, :), 1)/(t);%按照LE算法的写法
                S(i, j) = exp(-S(i, j));
            else
                S(i, j) = 0;
            end
            S(j, i) = S(i, j);
        end
    end
    Graph = S;
    %下面就是LE的步骤了
%     D = sum(S);
%     D = diag(D);
%     L = D - S;
%     L=max(L ,L');
%     [eigvector, eigvalue] = eig(L, D);%计算特征向量
%     [eigvalue, ansind] = sort(diag(eigvalue));
%     eigvector = eigvector(:,ansind(2:dis+1));
%     save('eigvector.mat', 'eigvector');
%     mappedX = eigvector;
    