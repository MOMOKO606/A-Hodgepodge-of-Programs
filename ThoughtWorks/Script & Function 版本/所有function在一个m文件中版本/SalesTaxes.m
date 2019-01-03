function SalesTaxes()
%  �������ܣ�Prints out the receipt details for these shopping lists.

%  ��������
[A,C,filename]=SalesTaxesInput();
%  ����
C=ComputeTaxes(C);
%  ������
OutputTaxes(A,C,filename);


function [A,C,filename]=SalesTaxesInput()
%  �������ܣ�����Ļ���������ĵ�����(����input1.txt)���������ݡ�
%  ���A������ԭʼ�����ĵ������ݣ�����ã�
%  ���filename������ã�
%  ���C������ͳ�ƺ�����ݣ���������ʹ�ã�

filename=input('�����룺�����ĵ�����(����input1.txt)��','s');
n=0;
i=1;
fid=fopen(filename);
if ( fid == -1 )
    errordlg('�Ҳ��������ļ�','������ʾ');
    return
end
while ( ~feof(fid) )
    A{i,1}=fgets(fid);  % ���ļ���ÿһ�ж���cell����MATLAB��cell���������ƽṹ��
    i=i+1;
    n=n+1;
end
fclose(fid);
C=CountC(A,n);


function C=CountC(A,n)
%  �������ܣ���cell����Aͳ�Ƶõ����C
%  C���кű�ʾ��Ʒ��ţ������i�б�ʾ��i����Ʒ��
%  C���кŷֱ��ʾ����˰�ʣ�����˰�ʣ�����������˰ǰ���ۣ����磺
%  -------------------------------------------
%  | ����˰�� | ����˰�� | �������� | ˰ǰ���� |
%  -------------------------------------------
%  ����'����˰��'Ϊ0��ʾ����Ʒ����book, food, medical product��'����˰��'Ϊ0.1��ʾ������Ʒ��
%  ������'����˰��'Ϊ0�ļ��Ͻ������ˡ�book��,'chocolate','pills'��
%  '����˰��'Ϊ0.1�ļ���������'CD','perfume'��
%  '����˰��'�н�����Ʒ˰��Ϊ0.05���ǽ�����Ʒ˰��Ϊ0��
%  �����߿���GetAttribt��������������'����˰��'�������Լ�ÿ��'����˰��'�а�������Ʒ��

for i=1:n
    str=char(A{i,1});  % ��cellת����char
    str=deblank(str);  %  ȥ����β�Ķ���ո�
    temp=regexp(str,'\s','split');  % �����а��ո񻮷�Ϊ���ɸ�����
    %  ��C��ֵ
    C(i,1)=GetAttribt(str);
    C(i,2)=IsImported(str);
    t1=str2num(char(temp{1,1}));
    t2=str2num(char(temp{1,length(temp)}));
    %  ��������������ʱ����ʾ����ֹ����
    if ( isempty(t1) || isempty(t2) )
        errordlg('�����ļ���ʽ��������������','������ʾ');
        return
    end
    C(i,3)=t1;
    C(i,4)=t2;
end


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


function C=ComputeTaxes(C)
%  �������ܣ�����table C�����嵥��������Ʒ��˰Ǯ��

[n,~]=size(C);
for i=1:n
    t=C(i,1)+C(i,2);
    t=t*C(i,4);
    p=floor(t/0.05);
    if ( p*0.05 ~= t )      
        t=0.05*(p+1);
    end
    C(i,5)=t*C(i,3);
end


function OutputTaxes(A,C,filename)
%  �������ܣ�������ս����

[n,~]=size(A);
SalesTaxes=0.0;
Total=0.0;
filename=['out',filename(3:length(filename))];
fid=fopen(filename,'w');
for i=1:n
    str=char(A{i,1});
    str=deblank(str);
    j=length(str);
    for k=1:2
        while( str(j) ~= ' ' )
            j=j-1;
        end
        j=j-1;
    end
    money=C(i,4)+C(i,5);
    Total=Total+money;
    money=[str(1:j),': ',num2str(money,'%.2f')];
    SalesTaxes=SalesTaxes+C(i,5);
    fprintf(fid,'%s\r\n',money);   
end
fprintf(fid,'%s\r\n',['Sales Taxes: ',num2str(SalesTaxes,'%.2f')]);
fprintf(fid,'%s\r\n',['Total: ',num2str(Total,'%.2f')]);
fclose(fid);