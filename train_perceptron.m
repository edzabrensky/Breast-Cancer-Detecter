function w = train_perceptron(input_x,input_y, w_init)
    w = w_init;
    numData = size(input_x(:,1),1);
    attributeSize = size(input_x(1,:),2);
    %set learning rate alpha
    alpha = .8;
    
    %for each data point
    for i=1:numData
            net =  0;
            %compute net
            for j=1:attributeSize
                net = net + input_x(i,j)*w(j,1);
            end
            %compute sigmoid of function
            net = 1.0/(1+exp(-net));
            %calculate error
            change = input_y(i, 1) - net;
            %adjust weight
           for j=1:attributeSize
                w(j,1) = w(j,1) + alpha*change*input_x(i,j);
           end
    end
      
end

