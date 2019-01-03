clear
clc

result=[];
m=input('Enter the number of cases:');
for i=1:m
    n=input('Enter the dimensions of the matrix:');
    A=input('Enter matrix A:');
    B=input('Enter matrix B:');
    YN=Dycrypt(A,B);
    result=strvcat(result,YN);
end
disp(result);