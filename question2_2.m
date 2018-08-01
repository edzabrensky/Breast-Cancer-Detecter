clc;clear;
a = load('breastcancerwinsconsin.mat'); 
data = [a.breastcancerwinsconsin1(:,(1:10)) a.breastcancerwinsconsin1(:,11)];

%shuffle rows 
data = data(randperm(size(data,1)),:);
xdata = data(:,(2:10));
ydata = data(:,11);

numFolds = 10;
[n,~] = size(xdata);
sizeTest = floor(n/numFolds);
ypred = zeros(sizeTest,1);
a=0;
accuracy = zeros(numFolds, 1);
sensitivity = zeros(numFolds, 1);
specificity = zeros(numFolds, 1);

%avg [p=1avg p=2avg p=1std p=2std] for each k average
avgAccuracy = zeros(numFolds, 4);
avgSensitivity = zeros(numFolds, 4);
avgSpecificity = zeros(numFolds, 4);
for k=1:10
    for p=1:2
        for w=1:numFolds
            if(w==1)
               xtest = xdata(1:sizeTest,:);
               xtrain = xdata(sizeTest+1:size(data),:);
               ytrain = ydata(sizeTest+1:size(data),:);
               ypred = knn_classifier(xtest, xtrain, ytrain, k, p);
            else
                a  =((w-1)*(sizeTest+1));
                xtest = xdata(a+1:a+sizeTest,:);
                xtrain = xdata(1:a,:);
                xtrain = [xtrain; xdata(a+sizeTest+1:n,:)];
                ytrain = ydata(1:a,:);
                ytrain = [ytrain; ydata(a+sizeTest+1:n,:)];
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
            %fprintf('fold %i: accuracy - %f, sensitivity - %f, specificity - %f\n',w, accuracy(w,1),sensitivity(w,1),specificity(w,1));
        end
        avgAccuracy(k,p) = mean(accuracy);
        avgAccuracy(k,p+2) = std(accuracy);
        avgSensitivity(k,p) = mean(sensitivity);
        avgSensitivity(k,p+2) = std(sensitivity);
        avgSpecificity(k,p) = mean(specificity);
        avgSpecificity(k,p+2) = std(specificity);
        fprintf('k:%i, p:%i \n', k,p);
        fprintf('Mean of accuracy: %f, Std: %f\n', mean(accuracy), std(accuracy));
        fprintf('Mean of sensitivity: %f, Std: %f\n', mean(sensitivity), std(sensitivity));
        fprintf('Mean of specificity: %f, Std: %f\n', mean(specificity), std(specificity));
    end
end

s = [1 2 3 4 5 6 7 8 9 10];
figure(1)
errorbar(s, avgAccuracy(:,1), avgAccuracy(:,3));
hold on
errorbar(s, avgAccuracy(:,2), avgAccuracy(:,4));
hold off
title('Number of nearest neighbors(k) vs Avg Accuracy');
xlabel('number of nearest neighbors(k)');
ylabel('Avg Accuracy');
legend({'p=1','p=2'}, 'Location', 'southeast');

figure(2)
errorbar(s, avgSensitivity(:,1),avgSensitivity(:,3));
hold on
errorbar(s, avgSensitivity(:,2),avgSensitivity(:,4));
hold off
title('Number of nearest neighbors(k) vs Avg sensitivity');
xlabel('number of nearest neighbors(k)');
ylabel('Avg sensitivity');
legend({'p=1','p=2'}, 'Location', 'southeast');

figure(3)
errorbar(s, avgSpecificity(:,1),avgSpecificity(:,3));
hold on
errorbar(s, avgSpecificity(:,2),avgSpecificity(:,4));
hold off
title('Number of nearest neighbors(k) vs Avg specificity');
xlabel('number of nearest neighbors(k)');
ylabel('Avg specificity');
legend({'p=1','p=2'}, 'Location', 'southeast');

