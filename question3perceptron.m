
clc;clear;
a = load('breastcancerwinsconsin.mat'); 
data = [a.breastcancerwinsconsin1(:,(1:10)) a.breastcancerwinsconsin1(:,11)];

%shuffle rows 
data = data(randperm(end),:);
xdata = data(:,(2:10));
ydata = data(:,11);

numFolds = 10;
[n,~] = size(xdata);
sizeTest = floor(n/numFolds);


%these matrices are here to store the avg  during each fold
%[avgaccuracyWhenW=0 avgAccuracyWhenW=Random]
accuracy = zeros(numFolds, 2);
sensitivity = zeros(numFolds, 2);
specificity = zeros(numFolds, 2);
avgAccuracy = zeros(10,4);
avgSensitivity = zeros(10,4);
avgSpecificity = zeros(10,4);
%for each fold
            
    for d=1:numFolds
        %when q=1 initialize w to 0 else to random
        for q=1:2
        %q=1;
        w = zeros(size(xdata(1,:), 2),1);

            %get xtrain and ytrain
            a  =((d-1)*(sizeTest+1));
            xtest = xdata(a+1:a+sizeTest,:);
            xtrain = xdata(1:a,:);
            xtrain = [xtrain; xdata(a+sizeTest+1:n,:)];
            ytrain = ydata(1:a,:);
            ytrain = [ytrain; ydata(a+sizeTest+1:n,:)];

            %convert y values
            y_input = zeros(size(ytrain,1),1);

              for i=1:n-sizeTest
                 if(ytrain(i,1) == 2)
                     y_input(i,1) = 1;
                 else
                     y_input(i,1) = 0;
                 end
              end
            %used for when q=2 so that we can get 10 independent runs of
            %the training so that randomness does not affect it
             avgWeight = zeros(9, 10);
             avgWeight(:,1) = w;
             if(q==2) 
                 %initialize w to random values from [-25,25]
                  for u=2:10
                      w = -25 + 50*rand(size(xdata,2),1);
                      avgWeight(:,u) = train_perceptron(xtrain,y_input,w);
                  end
                  for u=1:9
                      actualWeight(u,1) = mean(avgWeight(u,:),2);
                  end
                  w = actualWeight;       
             end
            w = train_perceptron(xtrain, y_input, w);


            %compute y label
            ypred = classify_perceptron(xtest,w);


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
            accuracy(d,q) = count/sizeTest;
            sensitivity(d,q) = truePositive/(truePositive+falseNegative);
            specificity(d,q) = trueNegative/(trueNegative+falsePositive);
        end
    end
    avgAccuracy(1,1) = mean(accuracy(:,1));
    avgAccuracy(1,3) = std(accuracy(:,1));
    avgAccuracy(1,2) = mean(accuracy(:,2));
    avgAccuracy(1,4) = std(accuracy(:,2));
    avgSensitivity(1,1) = mean(sensitivity(:,1));
    avgSensitivity(1,3) = std(sensitivity(:,1));
    avgSensitivity(1,2) = mean(sensitivity(:,2));
    avgSensitivity(1,4) = std(sensitivity(:,2));
    avgSpecificity(1,1) = mean(specificity(:,1));
    avgSpecificity(1,3) = std(specificity(:,1));
    avgSpecificity(1,2) = mean(specificity(:,2));
    avgSpecificity(1,4) = std(specificity(:,2));


s = [1];
figure(1)
errorbar(s, avgAccuracy(1,1), avgAccuracy(1,3));
hold on
errorbar(s, avgAccuracy(1,2), avgAccuracy(1,4));
hold off
title('W Initialization vs Avg Accuracy');
xlabel('W Initialization');
ylabel('Avg Accuracy');
legend({'w=0','w=random'}, 'Location', 'southeast');

figure(2)
errorbar(s, avgSensitivity(1,1),avgSensitivity(1,3));
hold on
errorbar(s, avgSensitivity(1,2),avgSensitivity(1,4));
hold off
title('W Initialization  vs Avg sensitivity');
xlabel('W Initialization');
ylabel('Avg sensitivity');
legend({'w=0','w=random'}, 'Location', 'southeast');

figure(3)
errorbar(s, avgSpecificity(1,1),avgSpecificity(1,3));
hold on
errorbar(s, avgSpecificity(1,2),avgSpecificity(1,4));
hold off
title('W Initialization vs Avg specificity');
xlabel('W Initialization');
ylabel('Avg specificity');
legend({'w=0','w=random'}, 'Location', 'southeast');

