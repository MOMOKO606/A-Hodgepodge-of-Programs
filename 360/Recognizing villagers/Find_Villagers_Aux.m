function [sum,T]=Find_Villagers_Aux(A,x,T)
sum=0;
[rows,~]=size(A);
for i=1:rows
    %  T(i)����1��ʾ�����Ѿ���ͳ�ƹ���ֱ�ӽ�����һ��
    if (T(i) == 0)
        %  ���A(i,1)Ϊx����A(i,2)Ϊx������
        %  ��sum��1�������ϵݹ�����A(i,2)���������
        if ( A(i,1) == x )
            T(i)=1;
            [y,T]=Find_Villagers_Aux(A,A(i,2),T);
            sum=sum+1+y;
        elseif ( A(i,2) == x )
            %  ���A(i,2)Ϊx����A(i,1)Ϊx������
            %  ��sum��1�������ϵݹ�����A(i,1)���������
            T(i)=1;
            [y,T]=Find_Villagers_Aux(A,A(i,1),T);
            sum=sum+1+y;
        end
    end
end

