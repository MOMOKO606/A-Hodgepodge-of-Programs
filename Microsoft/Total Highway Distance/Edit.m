function [q,d]=Edit(d,q,i,j,n,key)
%  ����THDֵ
delta=key-d(i,j);  %  �仯�Ĳ�ֵ
q=q+(n-i+j-2)*delta;
%  ����table d
for i=1:j-1
    d(i,j)=d(i,j)+delta;
end
for j=i+2:n
    d(i,j)=d(i,j)+delta;
end