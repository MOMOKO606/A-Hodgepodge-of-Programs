function Line_editor()
n=input('');  %  从屏幕读取n
A=cell(n,1);  %  创建n×1的cell A
%  将n行字符存入A中
for i=1:n
    A(i,1)={input('','s')};
end
for i=1:n
    string=char(A(i,1));
    S=[];
    Stop=0;
    for j=1:length((string))
        x=string(j);
        if (strcmp(x,'#'))
            [y,Stop]=POP(S,Stop);
        elseif (strcmp(x,'@'))
            while ( ~StackEmpty(Stop) )
                [y,Stop]=POP(S,Stop);
            end
        else
            [S,Stop]=PUSH(S,Stop,x);
        end
    end
    disp(char(S(1:Stop)));
end
        