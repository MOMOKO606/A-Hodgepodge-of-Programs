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
    C(i,3)=str2num(char(temp{1,1}));
    C(i,4)=str2num(char(temp{1,length(temp)}));
    if ( isempty(C(i,3)) || isempty(C(i,4)) )
        errordlg('输入文件格式有误，请重新输入','错误提示');
        return
    end
end