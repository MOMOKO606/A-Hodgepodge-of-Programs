function y=Sum_Matrix(A,ri,rj,ci,cj)
y=0;
for i=ri:rj
    for j=ci:cj
        y=y+A(i,j);
    end
end