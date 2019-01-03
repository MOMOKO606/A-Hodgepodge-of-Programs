function B=MatrixRotate_k(A,k,ri,rj,ci,cj)
%  函数功能：将 A[ri,ci],…，A[ri,cj] 这一部分矩阵顺时针旋转k*90°
%              A[rj,ci],…，A[rj,cj]
%  k=0,1,2,3

for i=1:k
    A=MatrixRotate(A,ri,rj,ci,cj);
end
B=A;
