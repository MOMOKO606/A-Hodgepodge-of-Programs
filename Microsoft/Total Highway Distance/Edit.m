function [q,d]=Edit(d,q,i,j,n,key)
%  更新THD值
delta=key-d(i,j);  %  变化的差值
q=q+(n-i+j-2)*delta;
%  更新table d
for i=1:j-1
    d(i,j)=d(i,j)+delta;
end
for j=i+2:n
    d(i,j)=d(i,j)+delta;
end