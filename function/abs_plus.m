function [B]=abs_plus(A)
    B = abs(A) ;
    [C,D] = min(B) ;
    if C == 0
        B(D) = min(B(setdiff(1:size(A,1)*size(A,2),D))) ;
    end
end