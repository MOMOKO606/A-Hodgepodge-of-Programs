function y=IsImported(str)
%  �������ܣ�CountC�ĸ����������ж���Ʒ�Ƿ���ڣ�����������˰�ʡ�
%  �������strΪ�ַ������͡�
%  �������imported��Ϊ0.05������Ϊ0

flag=strfind(str,'imported');
if ( isempty(flag) )  
    y=0;
else
    y=0.05;
end