function B=MatrixRotate(A,ri,rj,ci,cj)
%  �������ܣ��� A[ri,ci],����A[ri,cj] ��һ���־���˳ʱ����ת90��
%              A[rj,ci],����A[rj,cj]

for i=ri:rj
    for j=ci:cj
        T(ri+j-ci,cj-i+ri)=A(i,j);
    end
end
for i=ri:rj
    for j=ci:cj
        A(i,j)=T(i,j);
    end
end
B=A;
