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