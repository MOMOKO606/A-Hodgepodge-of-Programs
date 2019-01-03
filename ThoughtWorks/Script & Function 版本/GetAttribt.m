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

    
