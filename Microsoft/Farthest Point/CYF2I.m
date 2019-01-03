function y=CYF2I(y0,b)
%  Circle Y Float point to Integral point
%  函数功能：根据圆心y坐标y0，转换小数坐标b到圆内最接近b的整数坐标y

if ( b > y0 )
    y=floor(b);
elseif ( b < y0 )
    y=ceil(b);
else  %  b == y0
    if ( y0 == round(y0) )
        y=y0;
    else
        y=[];
    end
end