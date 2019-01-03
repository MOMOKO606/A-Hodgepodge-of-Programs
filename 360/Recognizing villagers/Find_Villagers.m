function Find_Villagers()
result=[];
%  从屏幕输入数据，以0，0结束
while ( true )
    x=input('n,m=');
    n=x(1);
    m=x(2);
    if ( n == 0 && m == 0)
        break
    end
    %  计算数据存入A中
    for i=1:m
        A(i,:)=input('');
    end
    %  Temp为动态规划辅助数组
    for i=1:m
        Temp(i)=0;
    end
    %  递归计算（动态规划）
    [y,~]=Find_Villagers_Aux(A,1,Temp);
    result=[result;y];
end
%  输出结果
disp(result);