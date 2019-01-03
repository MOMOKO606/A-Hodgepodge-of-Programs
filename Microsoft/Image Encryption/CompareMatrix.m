function TF=CompareMatrix(A,B,ri,rj,ci,cj)
TF=1;
for i=ri:rj
    for j=ci:cj
        if ( A(i,j) ~= B(i,j) )
            TF=0;
            return
        end
    end
end
