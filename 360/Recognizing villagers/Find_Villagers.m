function Find_Villagers()
result=[];
%  ����Ļ�������ݣ���0��0����
while ( true )
    x=input('n,m=');
    n=x(1);
    m=x(2);
    if ( n == 0 && m == 0)
        break
    end
    %  �������ݴ���A��
    for i=1:m
        A(i,:)=input('');
    end
    %  TempΪ��̬�滮��������
    for i=1:m
        Temp(i)=0;
    end
    %  �ݹ���㣨��̬�滮��
    [y,~]=Find_Villagers_Aux(A,1,Temp);
    result=[result;y];
end
%  ������
disp(result);