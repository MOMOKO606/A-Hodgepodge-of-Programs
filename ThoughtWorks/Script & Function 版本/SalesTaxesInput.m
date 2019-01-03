function [A,C,filename]=SalesTaxesInput()
%  函数功能：从屏幕键入输入文档名称(例如input1.txt)并读入数据。
%  输出A，保存原始输入文档的内容，输出用；
%  输出filename，输出用；
%  输出C，保存统计后的数据，后续计算使用；

filename=input('请输入：输入文档名称(例如input1.txt)：','s');
n=0;
i=1;
fid=fopen(filename);
while ( ~feof(fid) )
    A{i,1}=fgets(fid);  % 将文件的每一行读入cell矩阵，MATLAB中cell的作用类似结构体
    i=i+1;
    n=n+1;
end
fclose(fid);
C=CountC(A,n);