function SalesTaxes()
%  函数功能：Prints out the receipt details for these shopping lists.

%  输入数据
[A,C,filename]=SalesTaxesInput();
%  计算
C=ComputeTaxes(C);
%  输出结果
OutputTaxes(A,C,filename);


function [A,C,filename]=SalesTaxesInput()
%  函数功能：从屏幕键入输入文档名称(例如input1.txt)并读入数据。
%  输出A，保存原始输入文档的内容，输出用；
%  输出filename，输出用；
%  输出C，保存统计后的数据，后续计算使用；

filename=input('请输入：输入文档名称(例如input1.txt)：','s');
n=0;
i=1;
fid=fopen(filename);
if ( fid == -1 )
    errordlg('找不到输入文件','错误提示');
    return
end
while ( ~feof(fid) )
    A{i,1}=fgets(fid);  % 将文件的每一行读入cell矩阵，MATLAB中cell的作用类似结构体
    i=i+1;
    n=n+1;
end
fclose(fid);
C=CountC(A,n);


function C=CountC(A,n)
%  函数功能：由cell矩阵A统计得到表格C
%  C的行号表示商品序号，例如第i行表示第i件商品；
%  C的列号分别表示基础税率，进口税率，购买数量，税前单价，例如：
%  -------------------------------------------
%  | 基础税率 | 进口税率 | 购买数量 | 税前单价 |
%  -------------------------------------------
%  其中'基础税率'为0表示：商品属于book, food, medical product；'基础税率'为0.1表示其他商品；
%  本题中'基础税率'为0的集合仅收纳了‘book’,'chocolate','pills'；
%  '基础税率'为0.1的集合收纳了'CD','perfume'；
%  '进口税率'中进口商品税率为0.05，非进口商品税率为0；
%  开发者可在GetAttribt函数中自行扩充'基础税率'的种类以及每种'基础税率'中包含的商品。

for i=1:n
    str=char(A{i,1});  % 从cell转换成char
    str=deblank(str);  %  去掉首尾的多余空格
    temp=regexp(str,'\s','split');  % 将该行按空格划分为若干个部分
    %  给C赋值
    C(i,1)=GetAttribt(str);
    C(i,2)=IsImported(str);
    t1=str2num(char(temp{1,1}));
    t2=str2num(char(temp{1,length(temp)}));
    %  当输入数据有误时，提示并终止程序
    if ( isempty(t1) || isempty(t2) )
        errordlg('输入文件格式有误，请重新输入','错误提示');
        return
    end
    C(i,3)=t1;
    C(i,4)=t2;
end


function y=GetAttribt(str)
%  函数功能：CountC的辅助函数，判断商品属于哪一种基础税率。
%  输入参数str为字符串类型。
%  如果包含book,chocolate,pill则基础税率为0，否则为0.1

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
%  函数功能：CountC的辅助函数，判断商品是否进口，并给出进口税率。
%  输入参数str为字符串类型。
%  如果包含imported则为0.05，否则为0

flag=strfind(str,'imported');
if ( isempty(flag) )  
    y=0;
else
    y=0.05;
end


function C=ComputeTaxes(C)
%  函数功能：根据table C计算清单上所有商品的税钱。

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
%  函数功能：输出最终结果。

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