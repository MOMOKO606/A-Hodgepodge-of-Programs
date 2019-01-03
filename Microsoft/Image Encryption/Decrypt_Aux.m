function TF=Decrypt_Aux(A,B,ri,rj,ci,cj)
n=rj-ri+1;
if ( n == 1 )  %  base case
    if ( A(ri,ci) == B(ri,ci) )
        TF=1;
        return 
    else
        TF=0;
        return
    end
elseif ( mod(n,2) )  %  n为奇数时
    k=0;
    for i=1:4
        %  旋转A。（注意选择角度并不是单纯地k*90°，k=0,1,2,3）
        A=MatrixRotate_k(A,i-1-k,ri,rj,ci,cj);  
        k=i-1;  %  k表示之前A已旋转的次数
        TF=CompareMatrix(A,B,ri,rj,ci,cj);
        %  当旋转到某个角度时，A与B相等则直接返回TRUE
        if ( TF == 1 )
            return
        end
    end
else  %  n为偶数时
    index=QuarteredMatrix(ri,rj,ci,cj);  %  四等分矩阵B
    %  统计四等分后的B的每个sub矩阵的元素和
    for i=1:4
        Bsub(i)=Sum_Matrix(B,index(i,1),index(i,2),index(i,3),index(i,4));
    end
    Asub1=Sum_Matrix(A,index(1,1),index(1,2),index(1,3),index(1,4));
    k=0;
    for i=1:4
        if ( Asub1 == Bsub(i) )
             A=MatrixRotate_k(A,i-1-k,ri,rj,ci,cj);  
             k=i-1;
             FLAG=1;
             j=1;
             while ( FLAG && j <= 4 )
                 FLAG=Decrypt_Aux(A,B,index(j,1),index(j,2),index(j,3),index(j,4));
                 j=j+1;
             end
             if ( FLAG == 1 )
                 TF=1;
                 return
             end
        end
    end
    TF=0;
end
                     
                 
        
        
        
        
        
        
        
        
