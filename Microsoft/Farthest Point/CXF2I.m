function x=CXF2I(x0,a)
%  Circle X Float point to Integral point
%  �������ܣ�����Բ��x����x0��ת��С������a��Բ����ӽ�a����������x

if ( a > x0 )
    x=floor(a);
elseif ( a < x0 )
    x=ceil(a);
end