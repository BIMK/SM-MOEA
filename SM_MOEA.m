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
    thres = 0.1; %����ָ��˥������
    paretoAVE = zeros(1,2); %���������Pareto�Ľ��
    %[knnIndex] = preKNN(train_F);
    
    %% ��Ⱥ��ʼ������Ӧ�����ĵ�initializaition
    % ����ά�ȵ���Ӧ��
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
        dimFitness    = dimFitness + NDSort(dimfit, dim);  %ǰ���������Ϊά�ȵ���Ӧ��ֵ
    end
    % ������ʼ��Ⱥ
    %Dec = unifrnd(repmat(zeros(1,dim),sizep,1),repmat(ones(1,dim),sizep,1));ʵ������
    Dec = ones(sizep, dim);
    Mask = zeros(sizep,dim);
    for i = 1 : sizep
        Mask(i, TournamentSelection(2, ceil(rand*dim), dimFitness)) = 1; %����ά�ȵ���Ӧ��ֵ����Ӧ��λ����1��0
    end
    off = logical(Dec.*Mask);
    
    

    
    
    %% ���ۣ��ҳ���֧���
    for i=1:sizep
        [ofit(i,1),ofit(i,2)] = FSKNNfeixiang(off(i, :),train_F,train_L);
    end
    [FrontNO,~] = NDSort(ofit(:,1:2),sizep);
    site = find(FrontNO==1);
    solution = ofit(site,:);
    erBestParetoAVE = 1;  %����������ʷ����ֵ
    paretoAVE(1) = mean(solution(:,1));
    paretoAVE(2) = mean(solution(:,2));
    
    %% ά�Ȼ���Ϣ����MI����
    Y_train = train_L;
    %Y_train(Y_train==0)=-1;
    MI = zeros(1,dim);
    for i = 1:dim
        MI(i) = MItest(train_F(:,i),Y_train);
    end
    
    %% ����DRֵ
    DR = zeros(1,sizep);
    for i = 1:sizep
        DR(i) = sum(FrontNO>FrontNO(i));
    end
    DR = DR./sizep;
    
    %% ��ʼ��bitImportanceָ������
    bitImportance = zeros(sizep,dim);
    for i=1:sizep
        for j=1:dim
            bitImportance(i,j) = MI(j)*DR(i)/(sum(MI)*sizep);
        end
    end
    
    %% �м�������
    tAveError = zeros(1,maxFES);  %���д�����
    tAveFea = zeros(1,maxFES); %����������
    tErBest = zeros(1,maxFES);
    tThres = zeros(1,maxFES);
    tempVar = cell(1,4);
    
    %% ����
    while FES <= maxFES
        
        isChange = zeros(sizep,dim); %�ж��Ƿ�仯
        extTemp = 0; %������¼ÿ������ֵ�ľ�ֵ
        
        

        %----------------ά�ȱ���----------------
        for i = 1:sizep
            if(ismember(i,site)) %��pareto���ϵĽ�ȡ������
                continue;
            end
            
            curiOff = off(i,:); %��ǰ��i������, ������������
            curpSite = site(randi(length(site))); %���ѡ��һ��pareto���ϵĸ���
            pop = off(curpSite,:); %pareto���ϵ�һ������
            
            aveiBit = mean(bitImportance(i,:));
            
            for j = 1:dim
                popBit = pop(j);
                ext = 1/(1+exp(-5*(aveiBit-bitImportance(i,j))));
                tempThres = initThres * exp(-thres*FES);
                ext = ext * tempThres;
                if rand() < ext
                    off(i,j) = 0; %ά������
                end
                extTemp = extTemp + ext;
                
                %-----ά���˻�-----
                if bitImportance(i,j) > bitImportance(curpSite,j) %��ǰi����Ҫ�Ա����ŵ���Ҫ�Ը�
                    off(i,j) = curiOff(j); %�˻���ԭ�ȵ�ֵ
                else %��ǰi����Ҫ�Ա����ŵ���Ҫ�Ե�
                    if rand() < (bitImportance(curpSite,j) - bitImportance(i,j))/bitImportance(curpSite,j)
                        off(i,j) = popBit;
                    end
                end
                %-----ά���˻�-----
                
                
                %-----�жϴ�λ�Ƿ����-----
                if curiOff(j) ~= off(i,j)
                    isChange(i,j)=1;
                end
                %-----�жϴ�λ�Ƿ����-----

            end         
        end
        extTemp = extTemp/dim/sizep;
        tThres(FES) = extTemp;
        
        %----------------ά�ȱ���----------------
        
        
        
        
        %----------------��ʵ����----------------
        for i=1:sizep
            [ofit(i,1),ofit(i,2)] = FSKNNfeixiang(off(i,:),train_F,train_L);
         
        end
        [FrontNO,~] = NDSort(ofit(:,1:2),sizep);
        site = find(FrontNO==1);
        solution = ofit(site,:);
        oldERAVE = paretoAVE(1);
        paretoAVE(1) = mean(solution(:,1)); %������
        paretoAVE(2) = mean(solution(:,2));
        if paretoAVE(1) < erBestParetoAVE
            erBestParetoAVE = paretoAVE(1); %������ʷpareto���ŵ�ֵ
        end
        oldDR = DR;
        for i = 1:sizep
            DR(i) = sum(FrontNO>FrontNO(i));
        end
        DR = DR./sizep;
        %----------------��ʵ����----------------
        
        
        %-----�������-----
        if  paretoAVE(1) >= oldERAVE %����µ�err��ֵû�б��
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
                    if (FrontNO(i) <= FrontNO(tempIndex))  %������ڽϺõĸ���
                        if isChange(i,j) == 1 && off(i,j) == 1 %0��1
                            bitImportance(i,j)=bitImportance(i,j)*exp(0.8)^(1/sqrt(1+1.0));
                            bitImportance(i,j)=min(bitImportance(i,j),1.0);
                        elseif isChange(i,j) == 1 && off(i,j) == 0 %1��0
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
        %-----�������-----
        %----------------������ֵ�ĸ���----------------
        
        
        
        erBestID = find(ofit(:,1)==min(ofit(:,1)));
        erBestID = erBestID(1);
        TEMPCNT = ((cnti-1)*maxFES+FES)/CNTTIME;%���㷨�޹ؼ���ʣ��ʱ��
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