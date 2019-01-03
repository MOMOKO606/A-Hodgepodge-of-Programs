clear
clc

[A,C,filename]=SalesTaxesInput();
C=ComputeTaxes(C);
OutputTaxes(A,C,filename);





