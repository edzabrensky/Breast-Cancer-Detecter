%calculates distance from each test data point to all training data points
%sort those distances and get k first data objects, get the majority
%classification of those objects and return y_pred
%A=[distance label] for each test data calculate distance toward
%train data and store in first column
%sortrows(A,1) then choose k first data points and get mode of those
%values
%fill in y_pred matrix and start over until no more xtest data
function y_pred = knn_classifier(x_test, x_train, y_train, k, p)
    [sizeTest, ~] = size(x_test);%get size of x_test
    [sizeTrain, ~] = size(x_train);
    
    y_pred = zeros([sizeTest 1]);
    %calculate distance
    for i=1:sizeTest
       distanceBetweenAllPoints = zeros(sizeTrain, 2);
       distanceBetweenAllPoints(:,2) = y_train ;
       %calculate distance between x_test data point and all x_train
       for j=1:sizeTrain
           distanceBetweenAllPoints(j,1) = distance(x_test(i,:), x_train(j,:), p);
       end
       %sort distances
       distanceBetweenAllPoints = sortrows(distanceBetweenAllPoints);
       
       %get k nearest neighbors by grabbing first k data points, then use
       %mode on the vector for most frequent value
       distanceBetweenAllPoints = distanceBetweenAllPoints(1:k,2);
       y_pred(i,1) = mode(distanceBetweenAllPoints(1:k,1));
   end
    
end


