clear
clc

input('');
x0=ans(1);
y0=ans(2);
r=ans(3);

[x,y]=FarthestPoint(x0,y0,r);
disp([x,y]);

PrintCircle(x0,y0,r)