function y=CYF2I(y0,b)
%  Circle Y Float point to Integral point
%  �������ܣ�����Բ��y����y0��ת��С������b��Բ����ӽ�b����������y

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