function [train_F,train_L,test_F,test_L] = DIVDATA10fold(dataName,iter)
% the features of a data set is marked with "ins" , the label of a data set
% is marked with "lab"
file = ['dataset/',dataName,'.mat'];
load(file)
global fold
dataMat=ins;
len=size(dataMat,1);
maxV = max(dataMat);
minV = min(dataMat);
range = maxV-minV;
newdataMat = (dataMat-repmat(minV,[len,1]))./(repmat(range,[len,1]));
if mod(iter, 10) == 1
    Indices   =  crossvalind('Kfold', length(lab), fold);
    save Indices;
else
    load Indices;
end
if(mod(iter, 10) == 0)
    site = find(Indices == 10);
    test_F = newdataMat(site,:);
    test_L = lab(site);
    site2 = find(Indices ~= 10);
    train_F = newdataMat(site2,:);
    train_L =lab(site2);
else
    site = find(Indices == mod(iter,10));
    test_F = newdataMat(site,:);
    test_L = lab(site);
    site2 = find(Indices ~= mod(iter,10));
    train_F = newdataMat(site2,:);
    train_L =lab(site2);
end

end
