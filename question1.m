clc;clear;
a = load('breastcancerwinsconsin.mat'); 
data = a.breastcancerwinsconsin1;
xdata = data(:,(2:10));
ydata = data(:,11);

%get xtrain, xtest and ytrain
%80percent train 20% is test
[n,~] = size(xdata);
eighteyCutOff = floor(n*.8);
eighteyCutOffPlus1 = eighteyCutOff+1;

xtrain = xdata(1:eighteyCutOff,:);
xtest = xdata(eighteyCutOffPlus1:n,:);
ytrain = ydata(1:eighteyCutOff,:);

%test with k = 1 and p = 2
k = 1;
p = 2;
ypred = knn_classifier(xtest, xtrain, ytrain, k, p); 
sizeTest = n - eighteyCutOff;

%compute accuracy
count = 0;

for i=1:sizeTest
    if(ypred(i,1) == ydata(eighteyCutOff+i,1))
        count = count +1;
    end
    %fprintf('%i , %i\n', ypred(i,1), ydata(eighteyCutOff+i,1));
end
fprintf('%f',count/sizeTest);

