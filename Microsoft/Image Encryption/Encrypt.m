function A=Encrypt(B,ri,rj,ci,cj)
%  �������ܣ���B������ܣ����ܺ�ľ���ΪA

n=rj-ri+1;
if ( n ~= cj-ci+1 )
    disp('B should be square matrix!')
    return
end
if ( n == 1 )  %  base case nΪ1ʱֱ�ӷ��أ�������ת����Ϊ������ô��תֵ�����䣬�����壩��
    A=B;
    return
else
    A=MatrixRotate_k(B,round(3*rand),ri,rj,ci,cj);  %  ��ת
    if ( mod(n,2) == 0 )  %  ���nΪż�����ĵȷֺ������ת
        index=QuarteredMatrix(ri,rj,ci,cj);
        for i=1:4
            A=Encrypt(A,index(i,1),index(i,2),index(i,3),index(i,4));
        end
    end
end
    

