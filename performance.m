function [accurancy,sensitivity,specificity]=performance(group_original,group_final)

normal_index=find(group_original==0);
tumor_index=find(group_original==1);
predict_normal=group_final(normal_index);
predict_tumor=group_final(tumor_index);
group_normal=group_original(normal_index);
group_tumor=group_original(tumor_index);

P = group_tumor - predict_tumor;
FN=sum(~~P);
TP=length(P) - FN;
N = group_normal - predict_normal;
FP=sum(~~N);
TN=length(N) - FP;
P_Len=length(P);
N_Len=length(N);

accurancy = ( TP + TN )/( P_Len + N_Len);
sensitivity = TP / ( TP + FP );
specificity = TN / ( FN + TN );

end 