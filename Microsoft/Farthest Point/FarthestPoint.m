function [x,y]=FarthestPoint(x0,y0,r)
right=CXF2I(x0,x0+r);
left=CXF2I(x0,x0-r);
high=CXY2I(y0,y0+r);
low=CXY2I(y0,y0-r);
if ( left > right )
    disp('圆内不存在整数x坐标')
    return
end
if ( low > high )
    disp('圆内不存在整数y坐标')
    return
end
dmax=-inf;
for i=right:-1:left
    t=sqrt(r^2-(i-x0)^2);
    for j=1:2
        yin=CYF2I(y0,y0+t);
        %  出现坐标为(i,y0)且y0不是整数的点，则x继续向左前进
        if ( isempty(yin) ) 
            break
        end
        d=power((i-x0),2)+power((yin-y0),2);
        t=-t;
        if ( d > dmax )
            dmax=d;
            x=i;
            y=yin;
        elseif ( d == dmax )
            if ( i > x )
                x=i;
            elseif ( i == x )
                if ( yin > y )
                    y=yin;
                end
            end
        end
    end
end
        
        

    