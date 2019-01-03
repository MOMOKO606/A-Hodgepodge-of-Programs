clear
clc

B=round(200*rand(16,16));
A=Encrypt(B,1,16,1,16);
YN=Dycrypt(A,B);