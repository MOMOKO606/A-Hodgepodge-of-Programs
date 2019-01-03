function q=QuarteredMatrix(ri,rj,ci,cj)
%  函数功能：将由下标ri，rj，ci，cj表示的矩阵4等分。
%  ri表示输入矩阵的起始行；
%  rj表示输入矩阵的终止行；
%  ci表示输入矩阵的起始列；
%  cj表示输入矩阵的终止列；
%  将矩阵划分为：A11 A12  
%               A21 A12     
%  为了方便，进而表示为：
%               A1 A2
%               A4 A3 
%  将A1、A2、A3、A4的行列起始&终止下标依次存入q(i)中，i=1，2，3，4

n=rj-ri+1;
q(1,:)=[ri,ri+n/2-1,ci,ci+n/2-1];
q(2,:)=[ri,ri+n/2-1,ci+n/2,cj];
q(3,:)=[ri+n/2,rj,ci+n/2,cj];
q(4,:)=[ri+n/2,rj,ci,ci+n/2-1];
