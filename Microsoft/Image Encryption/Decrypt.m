function YN=Decrypt(A,B)
[n,m]=size(A);
if ( n ~= m)
    disp('A,B should be square matrix!');
    return
end
TF=Decrypt_Aux(A,B,1,n,1,n);
if ( TF == 1 )
    YN='Yes';
else
    YN='No';
end