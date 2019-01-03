function A=Encrypt(B,ri,rj,ci,cj)
%  函数功能：对B矩阵加密，加密后的矩阵为A

n=rj-ri+1;
if ( n ~= cj-ci+1 )
    disp('B should be square matrix!')
    return
end
if ( n == 1 )  %  base case n为1时直接返回，不再旋转（因为无论怎么旋转值都不变，无意义）。
    A=B;
    return
else
    A=MatrixRotate_k(B,round(3*rand),ri,rj,ci,cj);  %  旋转
    if ( mod(n,2) == 0 )  %  如果n为偶数，四等分后继续旋转
        index=QuarteredMatrix(ri,rj,ci,cj);
        for i=1:4
            A=Encrypt(A,index(i,1),index(i,2),index(i,3),index(i,4));
        end
    end
end
    

