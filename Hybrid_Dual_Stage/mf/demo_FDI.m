   clear;

opts.tol = 1e-1;
beta = 1:15;

z_ori = load('E:\MY\paper\FDILocation\code\data\case14\z_5-pbt.mat').z;
za = load('E:\MY\paper\FDILocation\code\data\case14\unmul\za_5-pbt2.mat').za;

A = zeros(100, 53);
Z = zeros(100, 53);
for i = 1:100
    t = [z_ori;z_ori];
    za_all = [t;za(i,:)];
%     if mod(i,2) == 0
%         za_all = [t;z_ori(i,:)];
%     else
%         za_all = [t;za(i,:)];
%     end
    [m,n] = size(za_all);
    k = 1;
    [X,Y,S,out] = lmafit_sms_v1(za_all,k,opts,beta);
    L = X*Y;
    A(i,:) = S(m,:);
    Z(i,:) = L(m,:);
end

save ('E:\MY\paper\FDILocation\code\data\case14\LRMF\a_new_um5-pbt2.mat',"A");
save ('E:\MY\paper\FDILocation\code\data\case14\LRMF\z_new_um10-pbt2.mat',"Z");

% imagesc(S,[-20 20]); 
% colormap("colorcube");

