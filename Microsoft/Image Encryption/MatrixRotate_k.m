function B=MatrixRotate_k(A,k,ri,rj,ci,cj)
%  �������ܣ��� A[ri,ci],����A[ri,cj] ��һ���־���˳ʱ����תk*90��
%              A[rj,ci],����A[rj,cj]
%  k=0,1,2,3

for i=1:k
    A=MatrixRotate(A,ri,rj,ci,cj);
end
B=A;
