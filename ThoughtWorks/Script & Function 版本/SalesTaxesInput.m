function [A,C,filename]=SalesTaxesInput()
%  �������ܣ�����Ļ���������ĵ�����(����input1.txt)���������ݡ�
%  ���A������ԭʼ�����ĵ������ݣ�����ã�
%  ���filename������ã�
%  ���C������ͳ�ƺ�����ݣ���������ʹ�ã�

filename=input('�����룺�����ĵ�����(����input1.txt)��','s');
n=0;
i=1;
fid=fopen(filename);
while ( ~feof(fid) )
    A{i,1}=fgets(fid);  % ���ļ���ÿһ�ж���cell����MATLAB��cell���������ƽṹ��
    i=i+1;
    n=n+1;
end
fclose(fid);
C=CountC(A,n);