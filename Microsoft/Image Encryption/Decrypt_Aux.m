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
elseif ( mod(n,2) )  %  nΪ����ʱ
    k=0;
    for i=1:4
        %  ��תA����ע��ѡ��ǶȲ����ǵ�����k*90�㣬k=0,1,2,3��
        A=MatrixRotate_k(A,i-1-k,ri,rj,ci,cj);  
        k=i-1;  %  k��ʾ֮ǰA����ת�Ĵ���
        TF=CompareMatrix(A,B,ri,rj,ci,cj);
        %  ����ת��ĳ���Ƕ�ʱ��A��B�����ֱ�ӷ���TRUE
        if ( TF == 1 )
            return
        end
    end
else  %  nΪż��ʱ
    index=QuarteredMatrix(ri,rj,ci,cj);  %  �ĵȷ־���B
    %  ͳ���ĵȷֺ��B��ÿ��sub�����Ԫ�غ�
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
                     
                 
        
        
        
        
        
        
        
        
