function C=ComputeTaxes(C)
%  �������ܣ�����table C�����嵥��������Ʒ��˰Ǯ��

[n,~]=size(C);
for i=1:n
    t=C(i,1)+C(i,2);
    t=t*C(i,4);
    p=floor(t/0.05);
    if ( p*0.05 ~= t )      
        t=0.05*(p+1);
    end
    C(i,5)=t*C(i,3);
end
