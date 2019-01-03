function index=Search(A,x)
n=length(A);
for i=1:n
    if ( A(i) == x )
        index=i;
        return
    end
end