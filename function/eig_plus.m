function [A_mat] = eig_plus(A_mat,cov_thresh)
%     A_mat = Sigma ;

%     %% 旧手法1
%     [V_mat,D_mat] = eig(A_mat) ;
%     if sum(diag(D_mat) < 0) > 0
%         A_mat = V_mat * abs(D_mat) / V_mat ;
%     end

%     %% 旧手法2
%     D_diag_vec = eig(A_mat) ;
%     if sum(D_diag_vec < 0) > 0
%         [V_mat,~] = eig(A_mat) ;
%         A_mat = V_mat * diag(abs(D_diag_vec)) / V_mat ;
%     end

    %% 新手法1
    if nargin == 1
        cov_thresh = 0.00001 ;
    end
    while min(real(eig(A_mat))) < cov_thresh
        A_mat = A_mat + cov_thresh * eye(size(A_mat,1)) ;
    end

end