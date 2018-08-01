function y_pred = classify_perceptron(input_x,w)
%CLASSIFY_PRECEPTRON Summary of this function goes here
%   Detailed explanation goes here
    sizeX = size(input_x(:,1),1);
    y_pred = zeros(sizeX,1);
    for m=1:sizeX
        sum = 0;
        for i=1:size(input_x(1,:),2)
            sum = sum + w(i,1)*input_x(m,i);
        end
        if(sign(sum) <0)
            y_pred(m,1) = 4;
        else
            y_pred(m,1) = 2;
        end
    end
end

