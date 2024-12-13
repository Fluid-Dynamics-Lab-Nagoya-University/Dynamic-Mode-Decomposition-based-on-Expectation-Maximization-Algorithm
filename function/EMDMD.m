function [coef_EMDMD,data_EMDMD,opt_EMDMD] = EMDMD(XX,r,flag_Sigma,data_name,opt_data_type,opt_conv)
%%%%% INPUT %%%%%%
% XX : Data
% r : Number of EMDMD modes
% flag_Sigma : Kind of EMDMD
% data_name : Data name to save
% opt_data_type : Data type setting
% opt_conv : Convergence setting
%%%%% OUTPUT %%%%%
% coef_EMDMD.A,C,Gamma,Sigma : EMDMD coefficients
% data_EMDMD.mu,muhat,V,Vhat : mean and covariance of state variables at Kalman filter and RTS smoother
% coef_EMDMD_ini.A,C,Gamma,Sigma : initial EMDMD coefficients
% opt_EMDMD.log_p : time history of log-likelihood
% opt_EMDMD.log_p : time history of log-likelihood
%%%%%%%%%%%%%%%%%%

    %% Setting
    mkdir output
    mkdir output/detail

    %% DMD
    % Change this section when using other DMD
    fprintf('Performing DMD\n')
    XX = set_data_type(XX,opt_data_type) ;
    X1 = XX(:,1:size(XX,2)-1) ;
    X2 = XX(:,2:size(XX,2)) ;
    m  = size(XX,1) ;   % Number of spatial components
    n  = size(XX,2) ;   % Number of temporal components
    [Usvd,Ssvd,Vsvd] = svd(X1,'econ') ; % SVD X1
    Usvd        = Usvd(:,1:r) ;         % Truncated U
    Ssvd        = Ssvd(1:r,1:r) ;       % Truncated S
    Vsvd        = Vsvd(:,1:r) ;         % Truncated V
    Atilde      = Usvd' * X2 * Vsvd * pinv(Ssvd) ; % Calculate A

    %% Initial setting of EMDMD
    fprintf('Performing initial setting of EMDMD\n')
    coef_EMDMD.A     = Atilde ;                                 % Initial value of A of EMDMD
    coef_EMDMD.C     = Usvd ;                                   % Initial value of C of EMDMD
    coef_EMDMD.Gamma = set_data_type(eye(r,r),opt_data_type) ;  % Initial value of Gamma (covariance of process noise) of EMDMD
    coef_EMDMD.Sigma = set_data_type(eye(m,m),opt_data_type) ;  % Initial value of Sigma (covariance of observation noise) of EMDMD
    clear X1 X2 Usvd Ssvd Vsvd Atilde;
    
    %% Prepare variables (= memory) during EMDMD
    SVsvd                  = pinv(coef_EMDMD.C) * XX ;
    data_EMDMD.mu          = set_data_type(zeros(r,n),opt_data_type) ;      % Mean of state variable during KF
    data_EMDMD.muhat       = set_data_type(zeros(r,n),opt_data_type) ;      % Mean of state variable during RTS smoother
    data_EMDMD.V           = set_data_type(zeros(r,r,n),opt_data_type) ;    % Covariance of state variable during KF
    data_EMDMD.Vhat        = set_data_type(zeros(r,r,n),opt_data_type) ;    % Covariance of state variable during RTS smoother
    data_EMDMD.muhat(:,1)  = SVsvd(:,1) ;  
    data_EMDMD.Vhat(:,:,1) = set_data_type(eye(r,r),opt_data_type) ;
    P        = set_data_type(zeros(r,r,n),opt_data_type) ;  % Variable essential for calculation
    J        = set_data_type(zeros(r,r,n),opt_data_type) ;  % Variable essential for calculation
    ln_d     = set_data_type(zeros(n,1),opt_data_type) ;    % Variable essential for calculation
    N        = set_data_type(zeros(n,1),opt_data_type) ;    % Variable essential for calculation
    Zero_r_r = set_data_type(zeros(r,r),opt_data_type) ;    % Constant essential for calculation
    Zero_r_m = set_data_type(zeros(r,m),opt_data_type) ;    % Constant essential for calculation
    I_r_r    = set_data_type(eye(r,r),opt_data_type) ;      % Constant essential for calculation
    I_m_m    = set_data_type(eye(m,m),opt_data_type) ;      % Constant essential for calculation    
    opt_EMDMD.log_p          = NaN([1 opt_conv.ite_max]) ;  % Time history of log-likelihood
    opt_EMDMD.log_p_accuracy = 0 ;                          % Number of failures that log-likelihood cannot be calculated 

    %% Save initial setting
    coef_EMDMD_ini       = coef_EMDMD ;
    data_EMDMD_ini.state = SVsvd ;
    data_EMDMD_ini.data  = XX ;
    save(['output/coef_EMDMD_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'_initial.mat'],'coef_EMDMD_ini','-v7.3') ;
    save(['output/data_EMDMD_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'_initial.mat'],'data_EMDMD_ini','-v7.3') ;
    clear XX SVsvd

    %% EMDMD
    if flag_Sigma ~= 2
        xnxn_for_C_Sigma = data_EMDMD_ini.data * data_EMDMD_ini.data' ;   % Constant essential for calculation
    end
    for iteration = 1:opt_conv.ite_max
        fprintf('Performing EMDMD at iteration=%d\n',iteration) ;

        %% Setting for faster calculation
        Eznzn_1_for_A_Gamma   = Zero_r_r ;  % Variable essential for calculation
        Ezn_1zn_1_for_A_Gamma = Zero_r_r ;  % Variable essential for calculation
        Eznzn_for_A_Gamma     = Zero_r_r ;  % Variable essential for calculation
        Eznxn_for_C_Sigma     = Zero_r_m ;  % Variable essential for calculation
        Eznzn_for_C_Sigma     = Zero_r_r ;  % Variable essential for calculation
        Sigma_inv     = const_diag(inv(coef_EMDMD.Sigma)) ;                     % Variable essential for calculation
        Sigma_inv_C   =                            Sigma_inv * coef_EMDMD.C ;   % Variable essential for calculation
        C_Sigma_inv   =            coef_EMDMD.C' * Sigma_inv ;                  % Variable essential for calculation
        C_Sigma_inv_C = const_diag(coef_EMDMD.C' * Sigma_inv * coef_EMDMD.C) ;  % Variable essential for calculation
        C_A           =            coef_EMDMD.C * coef_EMDMD.A ;                % Variable essential for calculation
        ln_det_Sigma  = 2*sum(log(diag(chol(coef_EMDMD.Sigma)))) ;              % Variable essential for calculation

        %% Prepare for Kalman filter and RTS smoother(k=0)
        mu_0     = data_EMDMD.muhat(:,1) ;
        P_0      = data_EMDMD.Vhat(:,:,1) ;
        L        = const_diag(Sigma_inv - Sigma_inv_C / (inv(P_0) + C_Sigma_inv_C) * C_Sigma_inv) ;
        K        = P_0 * coef_EMDMD.C' * L ;   
%         K        = P_0 * coef_EMDMD.C' * const_diag(Sigma_inv - Sigma_inv_C / (inv(P_0) + C_Sigma_inv_C) * C_Sigma_inv) ;   % when saving memory uncomment this code and comment out L, K, N
        data_EMDMD.mu(:,1)  = mu_0 + K * (data_EMDMD_ini.data(:,1) - coef_EMDMD.C * mu_0) ;
        data_EMDMD.V(:,:,1) = const_diag((I_r_r - K * coef_EMDMD.C) * P_0) ;
        ln_d1    = det(inv(P_0) + C_Sigma_inv_C) ;
        ln_d2    = det(P_0) ;
        if (ln_d1 < 0) || (ln_d2 < 0)
            ln_det_d1 = 2 * sum(log(diag(chol(eig_plus(inv(P_0) + C_Sigma_inv_C))))) ;
            ln_det_d2 = 2 * sum(log(diag(chol(eig_plus(P_0))))) ;
            opt_EMDMD.log_p_accuracy = opt_EMDMD.log_p_accuracy + 1 ;
        else
            ln_det_d1 = 2*sum(log(diag(chol(inv(P_0) + C_Sigma_inv_C)))) ;
            ln_det_d2 = 2*sum(log(diag(chol(P_0)))) ;
        end
        ln_d(1) = - 1/2 * (ln_det_d1 + ln_det_d2 + ln_det_Sigma) ;
        N(1)    = - 1/2 * (data_EMDMD_ini.data(:,1) - coef_EMDMD.C * mu_0)' * L * (data_EMDMD_ini.data(:,1) - coef_EMDMD.C * mu_0) ;
%         N(1)    = - 1/2 * (data_EMDMD_ini.data(:,1) - coef_EMDMD.C * mu_0)' * const_diag(Sigma_inv - Sigma_inv_C / (inv(P_0) + C_Sigma_inv_C) * C_Sigma_inv) * (data_EMDMD_ini.data(:,1) - coef_EMDMD.C * mu_0) ;   % when saving memory uncomment this code and comment out L, K, N
        
        %% Kalman filter(k=1,2,...,n)
        fprintf('Performing Kalman filter at iteration=%d\n',iteration) ;
        for k = 2:1:n
            P(:,:,k-1) = const_diag(coef_EMDMD.A * data_EMDMD.V(:,:,k-1) * coef_EMDMD.A' + coef_EMDMD.Gamma) ;
            L          = const_diag(Sigma_inv - Sigma_inv_C / (inv(P(:,:,k-1)) + C_Sigma_inv_C) * C_Sigma_inv) ;
            K          = P(:,:,k-1) * coef_EMDMD.C' * L ;
%             K          = P(:,:,k-1) * coef_EMDMD.C' * const_diag(Sigma_inv - Sigma_inv_C / (inv(P(:,:,k-1)) + C_Sigma_inv_C) * C_Sigma_inv) ;   % when saving memory uncomment this code and comment out L, K, N
            data_EMDMD.mu(:,k)  = coef_EMDMD.A * data_EMDMD.mu(:,k-1) + K * (data_EMDMD_ini.data(:,k) - C_A * data_EMDMD.mu(:,k-1)) ;
            data_EMDMD.V(:,:,k) = const_diag((I_r_r - K * coef_EMDMD.C) * P(:,:,k-1)) ;
            ln_d1 = det(inv(P(:,:,k-1)) + C_Sigma_inv_C) ;
            ln_d2 = det(P(:,:,k-1)) ;
            if (ln_d1 < 0) || (ln_d2 < 0)
                ln_det_d1 = 2 * sum(log(diag(chol(eig_plus(inv(P(:,:,k-1)) + C_Sigma_inv_C))))) ;
                ln_det_d2 = 2 * sum(log(diag(chol(eig_plus(P(:,:,k-1)))))) ;
                opt_EMDMD.log_p_accuracy = opt_EMDMD.log_p_accuracy + 1 ;
            else
                ln_det_d1 = 2 * sum(log(diag(chol(inv(P(:,:,k-1)) + C_Sigma_inv_C)))) ;
                ln_det_d2 = 2 * sum(log(diag(chol(P(:,:,k-1))))) ;
            end
            ln_d(k) = - 1/2 * (ln_det_d1 + ln_det_d2 + ln_det_Sigma) + ln_d(k-1) ;
            N(k)    = - 1/2 * (data_EMDMD_ini.data(:,k) - C_A * data_EMDMD.mu(:,k-1))' * L * (data_EMDMD_ini.data(:,k) - C_A * data_EMDMD.mu(:,k-1)) + N(k-1) ;
%             N(k)    = - 1/2 * (data_EMDMD_ini.data(:,k) - C_A * data_EMDMD.mu(:,k-1))' * const_diag(Sigma_inv - Sigma_inv_C / (inv(P(:,:,k-1)) + C_Sigma_inv_C) * C_Sigma_inv) * (data_EMDMD_ini.data(:,k) - C_A * data_EMDMD.mu(:,k-1)) + N(k-1) ; % 尤度計算用+メモリの使用量を節約する場合(L,K,Nをコメントアウト)
        end
        P(:,:,n) = const_diag(coef_EMDMD.A * data_EMDMD.V(:,:,n) * coef_EMDMD.A' + coef_EMDMD.Gamma) ;  % Not essential
        opt_EMDMD.log_p(iteration) = - n * m / 2 * log(2 * pi) + ln_d(n) + N(n) ;

        %% RTS smoother(k=n,n-1,...,1)
        fprintf('Performing RTS smoother at iteration=%d\n',iteration) ;
        J(:,:,n)    = data_EMDMD.V(:,:,n) * coef_EMDMD.A' / P(:,:,n) ;  % Not essential
        data_EMDMD.muhat(:,n)  = data_EMDMD.mu(:,n) ;
        data_EMDMD.Vhat(:,:,n) = data_EMDMD.V(:,:,n) ;
        for k = (n-1):-1:1
            J(:,:,k)    = data_EMDMD.V(:,:,k) * coef_EMDMD.A' / P(:,:,k) ;
            data_EMDMD.muhat(:,k)  = data_EMDMD.mu(:,k) + J(:,:,k) * (data_EMDMD.muhat(:,k+1) - coef_EMDMD.A * data_EMDMD.mu(:,k)) ;
            data_EMDMD.Vhat(:,:,k) = const_diag(data_EMDMD.V(:,:,k) + J(:,:,k) * (data_EMDMD.Vhat(:,:,k+1) - P(:,:,k)) * J(:,:,k)') ;
        end

        %% Update EMDMD coefficients
        fprintf('Updating EMDMD coefficients at iteration=%d\n',iteration) ;
        % Updata A and Gamma
        for k = 2:n
            Eznzn_1_for_A_Gamma   = Eznzn_1_for_A_Gamma   + data_EMDMD.muhat(:,k  ) * data_EMDMD.muhat(:,k-1)' + data_EMDMD.Vhat(:,:,k  ) * J(:,:,k-1)' ;
            Ezn_1zn_1_for_A_Gamma = Ezn_1zn_1_for_A_Gamma + data_EMDMD.muhat(:,k-1) * data_EMDMD.muhat(:,k-1)' + data_EMDMD.Vhat(:,:,k-1) ;
            Eznzn_for_A_Gamma     = Eznzn_for_A_Gamma     + data_EMDMD.muhat(:,k  ) * data_EMDMD.muhat(:,k  )' + data_EMDMD.Vhat(:,:,k  ) ;
        end
        Ezn_1zn_for_A_Gamma = Eznzn_1_for_A_Gamma' ;
        coef_EMDMD.A     = Eznzn_1_for_A_Gamma / Ezn_1zn_1_for_A_Gamma ;
        coef_EMDMD.Gamma = eig_plus(const_diag((Eznzn_for_A_Gamma - coef_EMDMD.A * Ezn_1zn_for_A_Gamma - Eznzn_1_for_A_Gamma * coef_EMDMD.A' + coef_EMDMD.A * Ezn_1zn_1_for_A_Gamma * coef_EMDMD.A') / (n-1))) ;

        % Updata C and Sigma
        for k = 1:n
%             xnxn_for_C_Sigma  = xnxn_for_C_Sigma  + data_EMDMD_ini.data(:,k) * data_EMDMD_ini.data(:,k)' ;  % Too slow
            Eznxn_for_C_Sigma = Eznxn_for_C_Sigma + data_EMDMD.muhat(:,k)    * data_EMDMD_ini.data(:,k)' ;
            Eznzn_for_C_Sigma = Eznzn_for_C_Sigma + data_EMDMD.muhat(:,k)    * data_EMDMD.muhat(:,k)' + data_EMDMD.Vhat(:,:,k) ;
        end
        xnEzn_for_C_Sigma = Eznxn_for_C_Sigma' ;
        coef_EMDMD.C = xnEzn_for_C_Sigma / Eznzn_for_C_Sigma ;

        if flag_Sigma == 0
%             coef_EMDMD.Sigma = (xnxn_for_C_Sigma - coef_EMDMD.C * Eznxn_for_C_Sigma - xnEzn_for_C_Sigma * coef_EMDMD.C' + coef_EMDMD.C * Eznzn_for_C_Sigma * coef_EMDMD.C') / n ;   % Accurate but comment out because unstable
            coef_EMDMD.Sigma = eig_plus(const_diag((xnxn_for_C_Sigma - coef_EMDMD.C * Eznxn_for_C_Sigma - xnEzn_for_C_Sigma * coef_EMDMD.C' + coef_EMDMD.C * Eznzn_for_C_Sigma * coef_EMDMD.C') / n)) ;
        elseif flag_Sigma == 1
            CC = coef_EMDMD.C' * coef_EMDMD.C ;
%             for k=1:n   % Too slow
%                 xnxn_for_Sigma     = xnxn_for_Sigma     + data_EMDMD_ini.data(:,k)' * data_EMDMD_ini.data(:,k) ;
%                 xnCEzn_for_Sigma   = xnCEzn_for_Sigma   + data_EMDMD_ini.data(:,k)' * coef_EMDMD.C * data_EMDMD.muhat(:,k) ;
%                 EznCCEzn_for_Sigma = EznCCEzn_for_Sigma + data_EMDMD.muhat(:,k)' * CC * data_EMDMD.muhat(:,k) + trace(CC * data_EMDMD.Vhat(:,:,k)) ;
%             end
            xnxn_for_Sigma     = sum(data_EMDMD_ini.data .^ 2,'all') ;
            xnCEzn_for_Sigma   = sum(coef_EMDMD.C * data_EMDMD.muhat .* data_EMDMD_ini.data,'all') ;
            EznCCEzn_for_Sigma = sum(diag(data_EMDMD.muhat' * CC * data_EMDMD.muhat),'all') + trace(CC * sum(data_EMDMD.Vhat,3)) ;
%             sigma = (xnxn_for_Sigma - 2 * xnCEzn_for_Sigma + EznCCEzn_for_Sigma) / (n * m) ;    % Accurate but comment out because unstable
            sigma = abs_plus((xnxn_for_Sigma - 2 * xnCEzn_for_Sigma + EznCCEzn_for_Sigma) / (n * m)) ;
            coef_EMDMD.Sigma = sigma * I_m_m ;
        elseif flag_Sigma == 2
%             for k=1:n   % Too slow
%                 x2_for_Sigma   = x2_for_Sigma   + data_EMDMD_ini.data(:,k) .^ 2 ;
%                 xCEz_for_Sigma = xCEz_for_Sigma + C(:,:) * muhat(:,k) .* data_EMDMD_ini.data(:,k) ;
%                 Ezc2_for_Sigma = Ezc2_for_Sigma + diag(C(:,:) * (Vhat(:,:,k) + muhat(:,k) * muhat(:,k)') * C(:,:)') ;
%             end
            x2_for_Sigma(:,1)   = sum(data_EMDMD_ini.data .^ 2,2) ;
            xCEz_for_Sigma(:,1) = sum(coef_EMDMD.C * data_EMDMD.muhat .* data_EMDMD_ini.data,2) ;
            Ezc2_for_Sigma(:,1) = diag(coef_EMDMD.C * (sum(data_EMDMD.Vhat,3) + data_EMDMD.muhat * data_EMDMD.muhat') * coef_EMDMD.C') ;
%             sigma = (x2_for_Sigma - 2 * xCEz_for_Sigma + Ezc2_for_Sigma) / n ;    % Accurate but comment out because unstable
            sigma = abs_plus((x2_for_Sigma - 2 * xCEz_for_Sigma + Ezc2_for_Sigma) / n) ;
            coef_EMDMD.Sigma = diag(sigma) ;
        elseif flag_Sigma == 3
            coef_EMDMD.Sigma = (xnxn_for_C_Sigma - coef_EMDMD.C * Eznxn_for_C_Sigma - xnEzn_for_C_Sigma * coef_EMDMD.C' + coef_EMDMD.C * Eznzn_for_C_Sigma * coef_EMDMD.C') / n ;
            sigma = diag(coef_EMDMD.Sigma) ;
            coef_EMDMD.Sigma = coef_EMDMD.Sigma - diag(sigma) ;
            % Avoid error arised from GPU by using CPU
            try
                [Usigma,Ssigma,Vsigma] = svd(coef_EMDMD.Sigma,'econ') ;
            catch
                save(['output/detail/coef_EMDMD_Sigma_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'_iter_',num2str(iteration),'.mat'],'coef_EMDMD.Sigma','-v7.3') ;
                [Usigma,Ssigma,Vsigma] = svd(gather(coef_EMDMD.Sigma),'econ') ;
                Usigma = set_data_type(Usigma,opt_data_type) ;
                Ssigma = set_data_type(Ssigma,opt_data_type) ;
                Vsigma = set_data_type(Vsigma,opt_data_type) ;                
            end
            coef_EMDMD.Sigma = eig_plus(const_diag(diag(sigma) - diag(diag(Usigma(:,1:r) * Ssigma(1:r,1:r) * Vsigma(:,1:r)')) + Usigma(:,1:r) * Ssigma(1:r,1:r) * Vsigma(:,1:r)')) ;
        end

        %% Saving
        save(['output/detail/coef_EMDMD_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'_iter_',num2str(iteration),'.mat'],'coef_EMDMD','-v7.3') ;
        save(['output/detail/data_EMDMD_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'_iter_',num2str(iteration),'.mat'],'data_EMDMD','-v7.3') ;

        %% Check convergence
        if opt_conv.ite_min < iteration
            fprintf('Log-likelihood difference is %d\n',opt_EMDMD.log_p(iteration)-opt_EMDMD.log_p(iteration-1))
            if (abs(opt_EMDMD.log_p(iteration) - opt_EMDMD.log_p(iteration-1))) < opt_conv.ite_threshold
                break;
            end
        end
    end
    
    %% Saving data
    if opt_EMDMD.log_p_accuracy > 0
        fprintf('Failed to calculate log-likelihood %d times\n',opt_EMDMD.log_p_accuracy)
    end
    save(['output/coef_EMDMD_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'_converged.mat'],'coef_EMDMD','-v7.3') ;
    save(['output/data_EMDMD_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'_converged.mat'],'data_EMDMD','-v7.3') ;
    save(['output/opt_EMDMD_', data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'_converged.mat'],'opt_EMDMD' ,'-v7.3') ;
end