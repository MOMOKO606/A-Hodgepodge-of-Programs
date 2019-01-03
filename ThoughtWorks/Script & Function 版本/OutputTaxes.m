function OutputTaxes(A,C,filename)
%  函数功能：输出最终结果。

[n,~]=size(A);
SalesTaxes=0.0;
Total=0.0;
filename=['out',filename(3:length(filename))];
fid=fopen(filename,'w');
for i=1:n
    str=char(A{i,1});
    str=deblank(str);
    j=length(str);
    for k=1:2
        while( str(j) ~= ' ' )
            j=j-1;
        end
        j=j-1;
    end
    money=C(i,4)+C(i,5);
    Total=Total+money;
    money=[str(1:j),': ',num2str(money,'%.2f')];
    SalesTaxes=SalesTaxes+C(i,5);
    fprintf(fid,'%s\r\n',money);   
end
fprintf(fid,'%s\r\n',['Sales Taxes: ',num2str(SalesTaxes,'%.2f')]);
fprintf(fid,'%s\r\n',['Total: ',num2str(Total,'%.2f')]);
fclose(fid);
    
    

    

