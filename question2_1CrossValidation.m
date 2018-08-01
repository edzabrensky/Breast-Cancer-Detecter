clc;clear;
a = load('breastcancerwinsconsin.mat'); 
data = [a.breastcancerwinsconsin1(:,(1:10)) a.breastcancerwinsconsin1(:,11)];

%shuffle rows 
data = data(randperm(end),:);
xdata = data(:,(2:10));
ydata = data(:,11);

numFolds = 10;
[n,~] = size(xdata);
k = 1;
p = 2;
sizeTest = floor(n/numFolds);
ypred = zeros(sizeTest,1);
a=0;
accuracy = zeros(numFolds, 1);
sensitivity = zeros(numFolds, 1);
specificity = zeros(numFolds, 1);

for w=1:numFolds
    if(w==1)
       xtest = xdata(1:sizeTest,:);
       xtrain = xdata(sizeTest+1:n,:);
       ytrain = ydata(sizeTest+1:n,:);
       ypred = knn_classifier(xtest, xtrain, ytrain, k, p);
    else
        a  =((w-1)*(sizeTest+1));
        xtest = xdata(a+1:a+sizeTest,:);
        xtrain = xdata(1:a,:);
        xtrain = [xtrain; xdata(a+sizeTest+1:n,:)];
        ytrain = ydata(1:a,:);
        ytrain = [ytrain; ydata(a+sizeTest+1:n,:)];
        %fprintf('test range: %i -%i, train trange: 1-%i, %i-n', a+1,a+sizeTest,a,a+sizeTest+1);
        ypred = knn_classifier(xtest, xtrain, ytrain, k, p);
    end
    %compute statistics here
    %compute accuracy
    count = 0;
    truePositive = 0;
    trueNegative = 0;
    falsePositive = 0;
    falseNegative = 0;
    for i=1:sizeTest
        if(ypred(i,1) == ydata(a+i,1))
            count = count +1;
        end
        if(ypred(i,1) == 4 && ydata(a+i,1) == 4) 
            truePositive = truePositive + 1;
        end
        if(ypred(i,1) == 2 && ydata(a+i,1) == 2) 
            trueNegative = trueNegative + 1;
        end
        if(ypred(i,1) == 4 && ydata(a+i,1) == 2) 
            falsePositive = falsePositive + 1;
        end
        if(ypred(i,1) == 2 && ydata(a+i,1) == 4) 
            falseNegative = falseNegative + 1;
        end
         
    end
    accuracy(w,1) = count/sizeTest;
    sensitivity(w,1) = truePositive/(truePositive+falseNegative);
    specificity(w,1) = trueNegative/(trueNegative+falsePositive);
    fprintf('fold %i: accuracy - %f, sensitivity - %f, specificity - %f\n',w, accuracy(w,1),sensitivity(w,1),specificity(w,1));
end
fprintf('Mean of accuracy: %f, Std: %f\n', mean(accuracy), std(accuracy));
fprintf('Mean of sensitivity: %f, Std: %f\n', mean(sensitivity), std(sensitivity));
fprintf('Mean of specificity: %f, Std: %f\n', mean(specificity), std(specificity));





