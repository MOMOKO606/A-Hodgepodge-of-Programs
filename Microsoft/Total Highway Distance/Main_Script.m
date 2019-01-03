clear
clc

x=input('n,m=');
n=x(1);
m=x(2);
x=input('');
c(1)=x(1);
c(2)=x(2);
d(1,2)=x(3);
for i=3:n
    x=input('');
    c(i)=x(2);
    d(i-1,i)=x(3);
end
[THD,d]=QueryInitial(d,n);
result=[];
for p=1:m
    str=input('','s');
    if (str(1) == 'Q')
        result=[result;THD];
    elseif (str(1) == 'E')
        x=str2num(str(6:length(str)));
        i=Search(c,x(1));
        j=Search(c,x(2));
        [THD,d]=Edit(d,THD,i,j,n,x(3));
    else
        return
    end
end
disp(result);
        
        
    
  

