function [train_F,train_L,test_F,test_L] = DIVDATA37(dataName)
file = ['dataset/',dataName,'.mat'];
load(file)

dataMat=ins;
len=size(dataMat,1);
%πÈ“ªªØ
maxV = max(dataMat);
minV = min(dataMat);
range = maxV-minV;
newdataMat = (dataMat-repmat(minV,[len,1]))./(repmat(range,[len,1]));

Indices   =  crossvalind('Kfold', length(lab), 10);
site = find(Indices==1|Indices==2|Indices==3);
test_F = newdataMat(site,:);
test_L = lab(site);
site2 = find(Indices~=1&Indices~=2&Indices~=3);
train_F = newdataMat(site2,:);
train_L =lab(site2);
end