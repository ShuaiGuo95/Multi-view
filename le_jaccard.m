function mappedX = le_jaccard(X, Y, n0, dis)
    
    sumn = 0;
    %在每个特征内计算以下变量：
    allnear = [];%存储每个样本的近邻
    alllabel = [];%存储所有样本的label
    allnearlabel = [];%存储第i个样本所有近邻的标签
    allC = [];%allC(i)统计每个类型的近邻个数
    for i = 1:size(X, 2)
        i
        tempi = cell2mat(X(i));
        n(i) = size(tempi, 1);%每种特征的样本个数
        
        temps = pdist2(tempi, tempi);%计算二范数距离
        [dummy, ind] = sort(temps, 2);%计算近邻排序
        ind = ind(:,1:n0);%取除本身外的n0个样本作为近邻
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
        sumn = sumn + n(i);%样本总数
    end
    save('allnear.mat', 'allnear');
    save('alllabel.mat', 'alllabel');
    save('allnearlabel.mat', 'allnearlabel');
    save('allC.mat', 'allC');
    size(allnear)
    
    S = [];
    for i = 1:sumn
        i
        tempi = allnearlabel(i, :);
        for j = 1:sumn
              tempj = allnearlabel(j, :);
%             if ismember(mod(i-1, 3000)+1, tempj)%i是不是j的近邻
%                 if floor((i-1)/3000) == floor((j-1)/3000)%相同特征之间

                    if ismember(alllabel(i), allnearlabel(j, :)) && ismember(alllabel(j), allnearlabel(i, :)) %i的label存在于j的近邻当中，且j的label存在于i的近邻当中
%                         n1c = length(find(tempi(:)==alllabel(j)));
%                         n2c = length(find(tempj(:)==alllabel(i)));
%                         S(i, j) = 1 - (n1c + n2c)/(2.2*n0);
                        S(i, j) = norm(allC(i, :)-allC(j, :), 1)/(2);%按照LE算法的写法
                        S(i, j) = exp(-S(i, j));
                    else
                        S(i, j) = 0;
                    end
%                     if alllabel(i)==alllabel(j)
%                         n1c = length(find(tempi(:)==alllabel(j)));
%                         n2c = length(find(tempj(:)==alllabel(i)));
%                         S(i, j) = 1 - (n1c + n2c)/length(union(tempi, tempj));
%                         S(j, i) = S(i, j);
%                     else
%                         n1c = length(find(tempi(:)~=alllabel(j)));
%                         n2c = length(find(tempj(:)~=alllabel(i)));
%                         S(i, j) = (n1c + n2c)/length(union(tempi, tempj));
%                         S(j, i) = S(i, j);
%                     end
%                 else%不同特征之间
%                     if alllabel(i) == alllabel(j)%label相同
%                         S(i, j) = norm(allC(i, :)-allC(j, :), 1)/(n0*2);
%                         S(i, j) = -exp(-S(i, j));
%                     else
%                         S(i, j) = 0;
%                     end
%                 end
%             else
%                 S(i, j) = 0;
%             end
%             if alllabel(i)==alllabel(j)
%                 n1c = length(find(tempi(:)==alllabel(j)));
%                 n2c = length(find(tempj(:)==alllabel(i)));
%                 S(i, j) = 1 - length(intersect(n1c, n2c))/length(union(tempi, tempj));
%                 S(j, i) = S(i, j);
%             else
%                 n1c = length(find(tempi(:)~=alllabel(j)));
%                 n2c = length(find(tempj(:)~=alllabel(i)));
%                 S(i, j) = length(intersect(n1c, n2c))/length(union(tempi, tempj));
%                 S(j, i) = S(i, j);
%             end
        end
    end
    save('S.mat', 'S');
    %下面就是LE的步骤了
    D = sum(S);
    D = diag(D);
    L = D - S;
%     %added
%     L(isnan(L)) = 0; D(isnan(D)) = 0;
% 	L(isinf(L)) = 0; D(isinf(D)) = 0;
    L=max(L ,L');
%     L=sparse(L);
%     D=sparse(D);
%     options.disp = 0;
%     options.isreal = 1;
%     options.issym = 1;
%     %added
    [eigvector, eigvalue] = eig(L, D);%计算特征向量
    [eigvalue, ansind] = sort(diag(eigvalue));
%     eigvector = eigvector';
    eigvector = eigvector(:,ansind(2:dis+1));
    
    %mapping = eigvector;
    save('eigvector.mat', 'eigvector');
    mappedX = eigvector;
    