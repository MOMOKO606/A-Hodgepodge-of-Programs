function [x,y]=FarthestPoint(x0,y0,r)
right=CXF2I(x0,x0+r);
left=CXF2I(x0,x0-r);
high=CXY2I(y0,y0+r);
low=CXY2I(y0,y0-r);
if ( left > right )
    disp('Բ�ڲ���������x����')
    return
end
if ( low > high )
    disp('Բ�ڲ���������y����')
    return
end
dmax=-inf;
for i=right:-1:left
    t=sqrt(r^2-(i-x0)^2);
    for j=1:2
        yin=CYF2I(y0,y0+t);
        %  ��������Ϊ(i,y0)��y0���������ĵ㣬��x��������ǰ��
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
        
        

    