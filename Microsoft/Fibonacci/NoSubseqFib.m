function result=NoSubseqFib(A,n)
%  Compute the number of non-empty sub-sequence of A which is a prefix of
%  Fibonacci

%  ����������ֵ
result=0;
%  Fibonacci���и���ֵ
j=1;
F(1)=1;
F(2)=1;
%  temp1���鸳��ֵ
k=1;
temp1(k)=0;
%  C�ĳ�ֵΪ0
C=zeros(n,n);  
%  �ڵ�j�У���q��AԪ����F(j)��ͬ��
%  ���Ե�q����0��ʱ�򣬱�ʾ�Ѿ�û��subsequence��prefix of F(j)��ȣ���ʱ�ѱ������е�prefix of
%  Fibonacci sequence��
%  q��ֵΪ1�����壬��Ϊ�˳���while�ܹ����С�
q=1;
%  while loopÿѭ��һ�α�ʾj������1����table C�����ƶ�һ�У�
%  temp1����Ϊk�������һ��loop����f(j-1)��ȵ�a(i)��index��
%  temp2����Ϊq����ű���loop����f(j)��ȵ�a(i)��index��
while( q > 0 )
    q=0;  
    temp2=[];
    stop=n;
    Maxtemp2=-inf;
    %  ����Fibonacci����Ԫ��
    if (j >= 3)
        F(j)=F(j-1)+F(j-2);
    end
    for p=1:k
        start=temp1(p);
        if (start < n)
            for i=start+1:stop
                if (A(i) == F(j))
                    if (start == 0 && j == 1)  %  j=1ʱ���������
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
             