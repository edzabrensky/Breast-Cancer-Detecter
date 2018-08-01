%calculates distance function
function dis = distance(x, y, p)
    [~,n] = size(x);%get size of vector
    sum = 0;
    %calculate distance
    for i=1:n
       sum = sum + (abs(x(1,i) - y(1,i)))^p; 
    end
    dis = sum^(1/p);
end


