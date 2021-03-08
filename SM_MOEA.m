function [solution, time, off, ofit, site, paretoAVE,tempVar,bitImportance] = SM_MOEA(train_F,train_L,cnti)
    fprintf('SM_MOEA 01\n');                                      
    tic
    global maxFES
    global knnIndex
    global sizep
    global CNTTIME
    FES = 1;
    dim = size(train_F,2);
    ofit = zeros(sizep,2);
    initThres = 1;
    thres = 0.1; %特征指数衰减常数
    paretoAVE = zeros(1,2); %保存的最终Pareto的结果
    %[knnIndex] = preKNN(train_F);
    
    %% 种群初始化，对应于论文的initializaition
    % 计算维度的适应度
    TDec    = []; 
    Tobj = []; 
    TMask   = [];
    TempPop = [];
    dimFitness = zeros(1,dim);
    
    for i = 1 : 1
        %Dec = unifrnd(repmat(zeros(1, dim),dim,1),repmat(ones(1, dim),dim,1));
        Dec = ones(dim, dim);
        Mask = eye(dim);
        pop = Dec.*Mask;
        TDec = [TDec;Dec];
        TMask      = [TMask;Mask];
        TempPop    = [TempPop;pop];
        dimfit = zeros(dim, 2);
        for m=1:dim
            [dimfit(m,1),dimfit(m,2)] = FSKNNfeixiang(pop(m, :),train_F,train_L);
        end
        ofit = fliplr(ofit);
        Tobj = [Tobj; ofit];
        dimFitness    = dimFitness + NDSort(dimfit, dim);  %前沿面序号作为维度的适应度值
    end
    % 产生初始种群
    %Dec = unifrnd(repmat(zeros(1,dim),sizep,1),repmat(ones(1,dim),sizep,1));实数创建
    Dec = ones(sizep, dim);
    Mask = zeros(sizep,dim);
    for i = 1 : sizep
        Mask(i, TournamentSelection(2, ceil(rand*dim), dimFitness)) = 1; %根据维度的适应度值给相应的位置置1或0
    end
    off = logical(Dec.*Mask);
    
    

    
    
    %% 评价，找出非支配解
    for i=1:sizep
        [ofit(i,1),ofit(i,2)] = FSKNNfeixiang(off(i, :),train_F,train_L);
    end
    [FrontNO,~] = NDSort(ofit(:,1:2),sizep);
    site = find(FrontNO==1);
    solution = ofit(site,:);
    erBestParetoAVE = 1;  %用来保留历史最优值
    paretoAVE(1) = mean(solution(:,1));
    paretoAVE(2) = mean(solution(:,2));
    
    %% 维度互信息矩阵MI生成
    Y_train = train_L;
    %Y_train(Y_train==0)=-1;
    MI = zeros(1,dim);
    for i = 1:dim
        MI(i) = MItest(train_F(:,i),Y_train);
    end
    
    %% 计算DR值
    DR = zeros(1,sizep);
    for i = 1:sizep
        DR(i) = sum(FrontNO>FrontNO(i));
    end
    DR = DR./sizep;
    
    %% 初始化bitImportance指导矩阵
    bitImportance = zeros(sizep,dim);
    for i=1:sizep
        for j=1:dim
            bitImportance(i,j) = MI(j)*DR(i)/(sum(MI)*sizep);
        end
    end
    
    %% 中间量保存
    tAveError = zeros(1,maxFES);  %所有错误率
    tAveFea = zeros(1,maxFES); %所有特征数
    tErBest = zeros(1,maxFES);
    tThres = zeros(1,maxFES);
    tempVar = cell(1,4);
    
    %% 迭代
    while FES <= maxFES
        
        isChange = zeros(sizep,dim); %判断是否变化
        extTemp = 0; %用来记录每代的阈值的均值
        
        

        %----------------维度变异----------------
        for i = 1:sizep
            if(ismember(i,site)) %对pareto面上的解取消变异
                continue;
            end
            
            curiOff = off(i,:); %当前第i个个体, 保存用来调整
            curpSite = site(randi(length(site))); %随机选择一个pareto面上的个体
            pop = off(curpSite,:); %pareto面上的一个个体
            
            aveiBit = mean(bitImportance(i,:));
            
            for j = 1:dim
                popBit = pop(j);
                ext = 1/(1+exp(-5*(aveiBit-bitImportance(i,j))));
                tempThres = initThres * exp(-thres*FES);
                ext = ext * tempThres;
                if rand() < ext
                    off(i,j) = 0; %维度置零
                end
                extTemp = extTemp + ext;
                
                %-----维度退化-----
                if bitImportance(i,j) > bitImportance(curpSite,j) %当前i的重要性比最优的重要性高
                    off(i,j) = curiOff(j); %退化至原先的值
                else %当前i的重要性比最优的重要性低
                    if rand() < (bitImportance(curpSite,j) - bitImportance(i,j))/bitImportance(curpSite,j)
                        off(i,j) = popBit;
                    end
                end
                %-----维度退化-----
                
                
                %-----判断此位是否变了-----
                if curiOff(j) ~= off(i,j)
                    isChange(i,j)=1;
                end
                %-----判断此位是否变了-----

            end         
        end
        extTemp = extTemp/dim/sizep;
        tThres(FES) = extTemp;
        
        %----------------维度变异----------------
        
        
        
        
        %----------------真实评价----------------
        for i=1:sizep
            [ofit(i,1),ofit(i,2)] = FSKNNfeixiang(off(i,:),train_F,train_L);
         
        end
        [FrontNO,~] = NDSort(ofit(:,1:2),sizep);
        site = find(FrontNO==1);
        solution = ofit(site,:);
        oldERAVE = paretoAVE(1);
        paretoAVE(1) = mean(solution(:,1)); %错误率
        paretoAVE(2) = mean(solution(:,2));
        if paretoAVE(1) < erBestParetoAVE
            erBestParetoAVE = paretoAVE(1); %保留历史pareto最优的值
        end
        oldDR = DR;
        for i = 1:sizep
            DR(i) = sum(FrontNO>FrontNO(i));
        end
        DR = DR./sizep;
        %----------------真实评价----------------
        
        
        %-----矩阵更新-----
        if  paretoAVE(1) >= oldERAVE %如果新的err均值没有变好
            oldBI = bitImportance;
            for i=1:sizep
                for j=1:dim
                    bitImportance(i,j) = 0.7*oldBI(i,j)+0.3*MI(j)*DR(i)/(sum(MI)*sizep);
                end
            end
        else
             for i=1:sizep
                for j=1:dim
                    tempIndex = site(randi(length(site)));
                    if (FrontNO(i) <= FrontNO(tempIndex))  %如果对于较好的个体
                        if isChange(i,j) == 1 && off(i,j) == 1 %0变1
                            bitImportance(i,j)=bitImportance(i,j)*exp(0.8)^(1/sqrt(1+1.0));
                            bitImportance(i,j)=min(bitImportance(i,j),1.0);
                        elseif isChange(i,j) == 1 && off(i,j) == 0 %1变0
                            bitImportance(i,j)=bitImportance(i,j)*exp(-0.2)^(1/sqrt(1+1.0));
                        end
                    else
                        %if DR(i)>oldDR(i)
                         %   bitImportance(i,j)=bitImportance(i,j)*exp(0.5)^(1/sqrt(1+1.0));
                          %  bitImportance(i,j)=min(bitImportance(i,j),1.0);
                       % else
                        %    bitImportance(i,j)=bitImportance(i,j)*exp(-0.5)^(1/sqrt(1+1.0));
                      %  end
                    end
                end
            end
        end
        %-----矩阵更新-----
        %----------------矩阵及阈值的更新----------------
        
        
        
        erBestID = find(ofit(:,1)==min(ofit(:,1)));
        erBestID = erBestID(1);
        TEMPCNT = ((cnti-1)*maxFES+FES)/CNTTIME;%与算法无关计算剩余时间
        fprintf('PRG: %.1f%%-- GEN: %2d  Error: %.5f  F: %.2f     ErBest: %.5f     thres: %.5f\n',100*TEMPCNT, FES,paretoAVE(1),paretoAVE(2),ofit(erBestID,1),extTemp);
        tAveError(FES) = paretoAVE(1);
        tAveFea(FES) = paretoAVE(2);
        tErBest(FES) = ofit(erBestID,1);
        FES = FES + 1;
    end
    %%
     
    [FrontNO,~] = NDSort(ofit(:,1:2),sizep);
    site = find(FrontNO==1);
    solution = ofit(site,:);
    
    paretoAVE(1) = mean(solution(:,1));
    paretoAVE(2) = mean(solution(:,2));
    tempVar{1} = tAveError;
    tempVar{2} = tAveFea;
    tempVar{3} = tErBest;
    tempVar{4} = tThres;
    clear tAveError;
    clear tAveFea;
    clear tErBest;
    clear tThres;
    toc
    time = toc;
 end