function x=CXF2I(x0,a)
%  Circle X Float point to Integral point
%  函数功能：根据圆心x坐标x0，转换小数坐标a到圆内最接近a的整数坐标x

if ( a > x0 )
    x=floor(a);
elseif ( a < x0 )
    x=ceil(a);
end