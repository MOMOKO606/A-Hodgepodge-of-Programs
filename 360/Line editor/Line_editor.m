function Line_editor()
n=input('');  %  ����Ļ��ȡn
A=cell(n,1);  %  ����n��1��cell A
%  ��n���ַ�����A��
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
        