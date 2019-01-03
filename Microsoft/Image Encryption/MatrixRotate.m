function B=MatrixRotate(A,ri,rj,ci,cj)
%  函数功能：将 A[ri,ci],…，A[ri,cj] 这一部分矩阵顺时针旋转90°
%              A[rj,ci],…，A[rj,cj]

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
