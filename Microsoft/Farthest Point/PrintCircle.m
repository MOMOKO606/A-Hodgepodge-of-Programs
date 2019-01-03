function PrintCircle(x0,y0,r)
x=[x0-r:0.005:x0+r];
t=sqrt(r^2-power((x-x0),2));
yhigh=y0+t;
ylow=y0-t;
plot(x,ylow,x,yhigh);
grid on