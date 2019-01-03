function [sum,d]=QueryInitial(d,n)
sum=0;
for l=2:n-1
    for i=1:n-l
        d(i,i+l)=d(i,i+l-1)+d(i+1,i+l);
        sum=sum+d(i,i+l);
    end
end
for i=1:n-1
    sum=sum+d(i,i+1);
end