function [error,fs] = FSKNNfeixiang(x,train_F,train_L)
global choice
inmodel = x>choice;%%%%%threshold
if(sum(inmodel)==0)
    error=1;
    fs=0;
    return;
end

train_f=train_F(:,inmodel);
[train_length,m] = size(train_f);
dis =zeros(train_length,train_length);
for i=1:train_length
    dis(i,i)=inf;
    for j =i+1:train_length
        d=0;
        for x=1:m
            d=d+ (train_f(i,x)-train_f(j,x))^2;
        end
        d=sqrt(d);
        dis(i,j)=d;
        dis(j,i)=d;
    end
    
end
error=0;
for i = 1:train_length
    [~,mindex] =min(dis(i,:));
    label = train_L(mindex);
    if(train_L(mindex) ~= train_L(i) )
        error= error+1;
    end
end
error=error/train_length;
fs = sum(inmodel);

end