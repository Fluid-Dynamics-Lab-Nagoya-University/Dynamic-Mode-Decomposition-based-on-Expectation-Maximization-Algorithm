function [XX,option] = make_data_vec_dis(m,n,meas_cov_vec,proc_cov_vec,dt,rng_num)
    mkdir output
    mkdir output/plot
    
    %% Setting
    rng(rng_num) ;
    init_cov = 0.1 ;        % initial noise
    
    % Eigenvalue
    f = [1.0 2.5 5.5] ;     % Real part of Eigenvalues
    g = [0   0   -0.3] ;    % Imaginary part of Eigenvalues
    k = 2 * length(f) ;	% (to perform DMD/TDMD with correct rank, set r=k)
    
    %% Generating data
    % Sysmte matrix A
    A1 = [] ;
    for ii = 1:length(f)
        A2 = [g(ii) 2*pi*f(ii); -2*pi*f(ii) g(ii)] ;
        A1 = [A1 A2] ;
    end
    Alowrank = [] ;
    for ii = 1:length(f)
        Alowrank = blkdiag(Alowrank,A1(:,(ii-1)*2+1:2*ii)) ;
    end
    option.true_evals = exp(eig(Alowrank) * dt) ;
    
    % Output matrix C
    [Q,~] = qr(randn(m,k),0) ;
    option.C = Q ;
    
    % Observation noise w
    sinmod = ones(m,n) + abs(repmat(randn(m,1),[1,n])) ;
    err    = meas_cov_vec .* sinmod ;
    
    % State variable x
    x0 = ones([k 1])+sqrt(init_cov).*randn(k,1) ;
    Xorg(:,1) = x0 ;
    Xsys(:,1) = x0 ;
    expA = expm(dt * Alowrank) ;
    option.A = expA ;
    for ii=2:n
        Xorg(:,ii) = expA * Xorg(:,ii-1) ;
        Xsys(:,ii) = expA * Xsys(:,ii-1) + sqrt(proc_cov_vec.*m/k) .* randn(k,1) ;
    end

    snapshots_org     = Q * Xorg ;
    snapshots_sys     = Q * Xsys ;
    snapshots_sys_obs = Q * Xsys + sqrt(err) .* randn(size(err)) ;

    XX            = snapshots_sys_obs(:,1:n) ;
    option.XX_sys = snapshots_sys(:,1:n) ;
    option.XX_org = snapshots_org(:,1:n) ;
    option.Sigma  = diag(err(:,1)) ;

    %% plot
    color_min = min(XX,[],'all') ;
    color_max = max(XX,[],'all') ;

    figure(1); hold on;
    imagesc(Xorg) ;
    colorbar ;
%     caxis([color_min color_max]) ;
    colorbar off ;
    xlim([0.5 size(Xorg,2)+0.5]) ;
    ylim([0.5 size(Xorg,1)+0.5]) ;
    ax = gca;
    outerpos = ax.OuterPosition;
    colormap('jet');
    left = outerpos(1);
    bottom = outerpos(2);
    ax_width = outerpos(3);
    ax_height = outerpos(4);
    ax.Position = [left bottom ax_width ax_height];
    xticks([])
    yticks([])
    xticklabels({})
    yticklabels({})
    saveas(gcf,['output/plot/data_var_wo_noise_',num2str(rng_num),'.fig'] ) ;
    saveas(gcf,['output/plot/data_var_wo_noise_',num2str(rng_num),'.png'] ) ;
    saveas(gcf,['output/plot/data_var_wo_noise_',num2str(rng_num),'.emf'] ) ;
    saveas(gcf,['output/plot/data_var_wo_noise_',num2str(rng_num),'.eps'],'epsc' ) ;

    figure(2); hold on;
    imagesc(Xsys) ;
    colorbar ;
%     caxis([color_min color_max]) ;
    colorbar off ;
    xlim([0.5 size(Xsys,2)+0.5]) ;
    ylim([0.5 size(Xsys,1)+0.5]) ;
    ax = gca;
    outerpos = ax.OuterPosition;
    colormap('jet');
    left = outerpos(1);
    bottom = outerpos(2);
    ax_width = outerpos(3);
    ax_height = outerpos(4);
    ax.Position = [left bottom ax_width ax_height];
    xticks([])
    yticks([])
    xticklabels({})
    yticklabels({})
    saveas(gcf,['output/plot/data_var_org_',num2str(rng_num),'.fig'] ) ;
    saveas(gcf,['output/plot/data_var_org_',num2str(rng_num),'.png'] ) ;
    saveas(gcf,['output/plot/data_var_org_',num2str(rng_num),'.emf'] ) ;
    saveas(gcf,['output/plot/data_var_org_',num2str(rng_num),'.eps'],'epsc' ) ;
    
    figure(3); hold on;
    imagesc(XX) ;
    colorbar ;
    caxis([color_min color_max]) ;
    colorbar off ;
    xlim([0.5 size(XX,2)+0.5]) ;
    ylim([0.5 size(XX,1)+0.5]) ;
    ax = gca;
    outerpos = ax.OuterPosition;
    colormap('jet');
    left = outerpos(1);
    bottom = outerpos(2);
    ax_width = outerpos(3);
    ax_height = outerpos(4);
    ax.Position = [left bottom ax_width ax_height];
    xticks([])
    yticks([])
    xticklabels({})
    yticklabels({})
    saveas(gcf,['output/plot/data_org_',num2str(rng_num),'.fig'] ) ;
    saveas(gcf,['output/plot/data_org_',num2str(rng_num),'.png'] ) ;
    saveas(gcf,['output/plot/data_org_',num2str(rng_num),'.emf'] ) ;
    saveas(gcf,['output/plot/data_org_',num2str(rng_num),'.eps'],'epsc' ) ;

    figure(4); hold on;
    imagesc(option.XX_sys) ;
    colorbar ;
    caxis([color_min color_max]) ;
    colorbar off ;
    xlim([0.5 size(XX,2)+0.5]) ;
    ylim([0.5 size(XX,1)+0.5]) ;
    ax = gca;
    outerpos = ax.OuterPosition;
    colormap('jet');
    left = outerpos(1);
    bottom = outerpos(2);
    ax_width = outerpos(3);
    ax_height = outerpos(4);
    ax.Position = [left bottom ax_width ax_height];
    xticks([])
    yticks([])
    xticklabels({})
    yticklabels({})
    saveas(gcf,['output/plot/data_org_wo_noise_',num2str(rng_num),'.fig'] ) ;
    saveas(gcf,['output/plot/data_org_wo_noise_',num2str(rng_num),'.png'] ) ;
    saveas(gcf,['output/plot/data_org_wo_noise_',num2str(rng_num),'.emf'] ) ;
    saveas(gcf,['output/plot/data_org_wo_noise_',num2str(rng_num),'.eps'],'epsc' ) ;
end