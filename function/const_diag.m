function [B] = const_diag(A)
    B = (A+A')/2 ;
end