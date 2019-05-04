function Graph = le_jaccard_graph(X, Y, n0, t)
    
    sumn = 0;
    %��ÿ�������ڼ������±�����
    allnear = [];%�洢ÿ�������Ľ���
    alllabel = [];%�洢����������label
    allnearlabel = [];%�洢��i���������н��ڵı�ǩ
    allC = [];%allC(i)ͳ��ÿ�����͵Ľ��ڸ���
    allmat = [];
    
    mindim = size(cell2mat(X(1)), 2);
    for i = 2:size(X, 2)
        mindim = min(mindim, size(cell2mat(X(i)), 2));
    end
    for i = 1:size(X, 2)
        i
        tempi = cell2mat(X(i));
        [tempi, ~] = apca(tempi, mindim);
        n(i) = size(tempi, 1);%ÿ����������������
        
        temps = pdist2(tempi, tempi);%�������������
        [~, ind] = sort(temps, 2);%�����������
        ind = ind(:,1:n0);%ȡǰn0��������Ϊ����
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
        sumn = sumn + n(i);%��������
    end
    
    S = zeros(sumn, sumn);
    for i = 1:sumn
        i
        tempi = allnearlabel(i, :);
        for j = i+1:sumn %i+1-->i
            tempj = allnearlabel(j, :);
            if ismember(alllabel(i), tempj) && ismember(alllabel(j), tempi) %i��label������j�Ľ��ڵ��У���j��label������i�Ľ��ڵ���
                S(i, j) = norm(allmat(i, :)-allmat(j, :), 1)/(t);%����LE�㷨��д��
                S(i, j) = exp(-S(i, j));
            else
                S(i, j) = 0;
            end
            S(j, i) = S(i, j);
        end
    end
    Graph = S;
    %�������LE�Ĳ�����
%     D = sum(S);
%     D = diag(D);
%     L = D - S;
%     L=max(L ,L');
%     [eigvector, eigvalue] = eig(L, D);%������������
%     [eigvalue, ansind] = sort(diag(eigvalue));
%     eigvector = eigvector(:,ansind(2:dis+1));
%     save('eigvector.mat', 'eigvector');
%     mappedX = eigvector;
    