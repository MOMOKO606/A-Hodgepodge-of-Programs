function [sum,T]=Find_Villagers_Aux(A,x,T)
sum=0;
[rows,~]=size(A);
for i=1:rows
    %  T(i)等于1表示该行已经被统计过，直接进入下一行
    if (T(i) == 0)
        %  如果A(i,1)为x，则A(i,2)为x的老乡
        %  则sum加1，并加上递归计算的A(i,2)的老乡个数
        if ( A(i,1) == x )
            T(i)=1;
            [y,T]=Find_Villagers_Aux(A,A(i,2),T);
            sum=sum+1+y;
        elseif ( A(i,2) == x )
            %  如果A(i,2)为x，则A(i,1)为x的老乡
            %  则sum加1，并加上递归计算的A(i,1)的老乡个数
            T(i)=1;
            [y,T]=Find_Villagers_Aux(A,A(i,1),T);
            sum=sum+1+y;
        end
    end
end

