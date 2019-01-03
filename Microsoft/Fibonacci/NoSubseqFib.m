function result=NoSubseqFib(A,n)
%  Compute the number of non-empty sub-sequence of A which is a prefix of
%  Fibonacci

%  计数器赋初值
result=0;
%  Fibonacci数列赋初值
j=1;
F(1)=1;
F(2)=1;
%  temp1数组赋初值
k=1;
temp1(k)=0;
%  C的初值为0
C=zeros(n,n);  
%  在第j列，有q个A元素与F(j)相同；
%  所以当q等于0的时候，表示已经没有subsequence与prefix of F(j)相等，此时已遍历所有的prefix of
%  Fibonacci sequence；
%  q初值为1无意义，仅为了初次while能够运行。
q=1;
%  while loop每循环一次表示j增加了1，即table C向右移动一列；
%  temp1长度为k，存放上一轮loop中与f(j-1)相等的a(i)的index；
%  temp2长度为q，存放本轮loop中与f(j)相等的a(i)的index。
while( q > 0 )
    q=0;  
    temp2=[];
    stop=n;
    Maxtemp2=-inf;
    %  计算Fibonacci序列元素
    if (j >= 3)
        F(j)=F(j-1)+F(j-2);
    end
    for p=1:k
        start=temp1(p);
        if (start < n)
            for i=start+1:stop
                if (A(i) == F(j))
                    if (start == 0 && j == 1)  %  j=1时的特殊情况
                        C(i,j)=C(i,j)+1;
                        result=result+1;
                    else
                        C(i,j)=C(i,j)+C(start,j-1);
                        result=result+C(start,j-1);
                    end
                    if ( i > Maxtemp2)
                        q=q+1;
                        temp2(q)=i;
                        Maxtemp2=i;
                    end
                end
            end
        end
        stop=Maxtemp2;
    end
    j=j+1;
    k=q;
    temp1=temp2;
end
             