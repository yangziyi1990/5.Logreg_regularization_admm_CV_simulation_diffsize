function [accurancy,sensitivity,specificity]=performance_beta(beta_true,beta_opt)

normal_index=find(beta_true~=0); % ¡¤?0, T
tumor_index=find(beta_true==0); % 0, F
predict_normal=beta_opt(normal_index);
predict_tumor=beta_opt(tumor_index);
group_normal=beta_true(normal_index);
group_tumor=beta_true(tumor_index);

predict_normal(predict_normal~=0)=1;
predict_tumor(predict_tumor~=0)=1;
group_normal(group_normal~=0)=1;
group_tumor(group_tumor~=0)=1;

P = group_normal - predict_normal;
FN=sum(~~P);
TP=length(P) - FN;
N = group_tumor - predict_tumor;
FP=sum(~~N);
TN=length(N) - FP;
P_Len=length(P);
N_Len=length(N);

accurancy = ( TP + TN )/( P_Len + N_Len);
sensitivity = TP / ( TP + FN );
specificity = TN / ( FP + TN );

end 