function y=GetAttribt(str)
%  �������ܣ�CountC�ĸ����������ж���Ʒ������һ�ֻ���˰�ʡ�
%  �������strΪ�ַ������͡�
%  �������book,chocolate,pill�����˰��Ϊ0������Ϊ0.1

compare={'book';'chocolate';'pill'};
for i=1:3
    flag=strfind(str,char(compare{i,1}));
    if ( isempty(flag) )
        flag=0;
    else 
        flag=1;
        break
    end
end
if ( flag == 1 )  
    y=0;
else
    y=0.1;
end

    
