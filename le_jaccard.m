function mappedX = le_jaccard(X, Y, n0, dis)
    
    sumn = 0;
    %��ÿ�������ڼ������±�����
    allnear = [];%�洢ÿ�������Ľ���
    alllabel = [];%�洢����������label
    allnearlabel = [];%�洢��i���������н��ڵı�ǩ
    allC = [];%allC(i)ͳ��ÿ�����͵Ľ��ڸ���
    for i = 1:size(X, 2)
        i
        tempi = cell2mat(X(i));
        n(i) = size(tempi, 1);%ÿ����������������
        
        temps = pdist2(tempi, tempi);%�������������
        [dummy, ind] = sort(temps, 2);%�����������
        ind = ind(:,1:n0);%ȡ���������n0��������Ϊ����
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
        sumn = sumn + n(i);%��������
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
%             if ismember(mod(i-1, 3000)+1, tempj)%i�ǲ���j�Ľ���
%                 if floor((i-1)/3000) == floor((j-1)/3000)%��ͬ����֮��

                    if ismember(alllabel(i), allnearlabel(j, :)) && ismember(alllabel(j), allnearlabel(i, :)) %i��label������j�Ľ��ڵ��У���j��label������i�Ľ��ڵ���
%                         n1c = length(find(tempi(:)==alllabel(j)));
%                         n2c = length(find(tempj(:)==alllabel(i)));
%                         S(i, j) = 1 - (n1c + n2c)/(2.2*n0);
                        S(i, j) = norm(allC(i, :)-allC(j, :), 1)/(2);%����LE�㷨��д��
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
%                 else%��ͬ����֮��
%                     if alllabel(i) == alllabel(j)%label��ͬ
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
    %�������LE�Ĳ�����
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
    [eigvector, eigvalue] = eig(L, D);%������������
    [eigvalue, ansind] = sort(diag(eigvalue));
%     eigvector = eigvector';
    eigvector = eigvector(:,ansind(2:dis+1));
    
    %mapping = eigvector;
    save('eigvector.mat', 'eigvector');
    mappedX = eigvector;
    