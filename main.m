clear all;
close all;
clc ;
addpath('function')

%% Parameters of EMDMD
% Reccomend using EMDMD with diagonal matrix constraint and whose diagonal components are not same
r_all      = 6 ;        % Number of EMDMD modes
flag_Sigma = 2 ;        % Type of EMDMD
% flag_Sigma = 0 : EMDMD without constraint
% flag_Sigma = 1 : EMDMD with diagonal matrix constraint and whose diagonal components are same
% flag_Sigma = 2 : EMDMD with diagonal matrix constraint and whose diagonal components are not same
% flag_Sigma = 3 : EMDMD with low-rank reconstruction
data_str   = 'art' ;    % Data name for saving

%% Machine setting
% Reccomend using single type and GPU when perfoming at large data
opt_data_type.flag_precision_type = 1 ;
% opt_data_type.flag_precision_type = 1 : single type
% opt_data_type.flag_precision_type = 2 : double type
opt_data_type.flag_processing_unit_type = 0 ;
% opt_data_type.flag_processing_unit_type = 0 : CPU
% opt_data_type.flag_processing_unit_type = 1 : GPU

%% Option convergence parameters of EMDMD
opt_conv.ite_min       = 10 ;   % Minimum number of iterations
opt_conv.ite_max       = 100 ;  % Maximum number of iterations
opt_conv.ite_threshold = 1 ;    % Threshold value of iterations

%% Data setting (for sample program)
m        = 100 ;        % Number of spatial components
n        = 1000 ;       % Number of temporal components
dt       = 0.01 ;       % Time interval
proc_cov = 0.000049 ;   % Strength of system noise (Not covariance)
meas_cov = 0.09 ;       % Strength of observation noise (Not covariance)
ense_max = 10 ;          % Number of ensembles

%% Core
for r_num = 1:length(r_all)
    r = r_all(r_num) ;
    for ense_num = 1:ense_max
        close all;

        %% Generating data with noise
        rng(ense_num) ;
        proc_cov_vec = proc_cov + proc_cov/10 * randn([6,1]) ;  % Covariance of system noise
        meas_cov_vec = meas_cov + meas_cov/10 * randn([m,1]) ;  % Covariance of observation noise
        [XX,opt_XX]  = make_data_vec_dis(m,n,meas_cov_vec,proc_cov_vec,dt,ense_num) ;
        data_name = [data_str,'_',num2str(ense_num)] ;
        save(['output/opt_data_',data_name,'.mat'],'opt_XX','-v7.3') ;

        %% EMDMD
        [coef_EMDMD,data_EMDMD,opt_EMDMD] = EMDMD(XX,r,flag_Sigma,data_name,opt_data_type,opt_conv) ;
        [Y,D]              = eig(coef_EMDMD.A) ;
        result_EMDMD.evals = diag(D) ;
        result_EMDMD.evecs = coef_EMDMD.C * Y ;
        result_EMDMD.XX    = coef_EMDMD.C * data_EMDMD.muhat ;
        result_EMDMD.err   = norm(result_EMDMD.XX - opt_XX.XX_sys,'fro') / norm(opt_XX.XX_sys,'fro') ;
        save(['output/result_EMDMD_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'.mat'],'result_EMDMD','-v7.3') ;

        %% DMD and POD (Target of comparison)
        % DMD is employed for comparing eigenvalues 
        % POD is employed for comparing reconstruction error
        load(['output/coef_EMDMD_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'_initial.mat']) ;
        load(['output/data_EMDMD_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'_initial.mat']) ;
        [Y,D]            = eig(coef_EMDMD_ini.A) ;
        result_DMD.evals = diag(D) ;
        result_DMD.evecs = coef_EMDMD_ini.C * Y ;
        result_POD.XX    = coef_EMDMD_ini.C * coef_EMDMD_ini.C' * data_EMDMD_ini.data ;
        result_POD.err   = norm(result_POD.XX - opt_XX.XX_sys,'fro') / norm(opt_XX.XX_sys,'fro') ;
        save(['output/result_DMD_',data_name,'_mode_',num2str(r),'.mat'],'result_DMD','-v7.3') ;
        save(['output/result_POD_',data_name,'_mode_',num2str(r),'.mat'],'result_POD','-v7.3') ;

    end
end

% Plot
for r_num = 1:length(r_all)
    close all;
    r = r_all(r_num) ;
    
    true_evals  = [] ;
    EMDMD_evals = [] ;
    DMD_evals   = [] ;
    EMDMD_err   = [] ;
    POD_err     = [] ;

    for ense_num = 1:ense_max
        data_name = [data_str,'_',num2str(ense_num)] ;
        load(['output/opt_data_',    data_name,'.mat']) ;
        load(['output/result_EMDMD_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'.mat']) ;
        load(['output/result_DMD_',  data_name,'_mode_',num2str(r),'.mat']) ;
        load(['output/result_POD_',  data_name,'_mode_',num2str(r),'.mat']) ;
        true_evals  = [true_evals;  eig(opt_XX.A)] ;
        EMDMD_evals = [EMDMD_evals; result_EMDMD.evals] ;
        DMD_evals   = [DMD_evals;   result_DMD.evals] ;
        EMDMD_err   = [EMDMD_err;   result_EMDMD.err] ;
        POD_err     = [POD_err;     result_POD.err] ;

        % Data
        load(['output/data_EMDMD_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'_initial.mat']) ;
        color_min = min(data_EMDMD_ini.data,[],'all') ;
        color_max = max(data_EMDMD_ini.data,[],'all') ;
        figure(ense_num+100); hold on;
        imagesc(result_EMDMD.XX) ;
        colorbar ;
        caxis([color_min color_max]) ;
        colorbar off ;
        xlim([0.5 size(result_EMDMD.XX,2)+0.5]) ;
        ylim([0.5 size(result_EMDMD.XX,1)+0.5]) ;
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
        saveas(gcf,['output/plot/data_EMDMD_wo_noise_',data_name,'_',num2str(ense_num),'.fig'] ) ;
        saveas(gcf,['output/plot/data_EMDMD_wo_noise_',data_name,'_',num2str(ense_num),'.png'] ) ;
        saveas(gcf,['output/plot/data_EMDMD_wo_noise_',data_name,'_',num2str(ense_num),'.emf'] ) ;
        saveas(gcf,['output/plot/data_EMDMD_wo_noise_',data_name,'_',num2str(ense_num),'.eps'],'epsc' ) ;

        % Specific component of data(vsPOD)
        x_dir = [0:n-1] ;
        comp_num = 50 ;
        figure(ense_num+200); hold on;
        title('Data (gray), data w/o noise (black), EMDMD (red), and POD (green)')
        box on ;
        set(gcf,'Position',[0,0,720,300]);
        set(gcf,'defaultAxesFontName','Arial');
        set(gcf,'defaultTextFontName','Arial');
        set(gcf,'defaultAxesFontSize',9);
        set(gcf,'defaultTextFontSize',9);
        set(gca,'FontSize',9);
        plot(x_dir,data_EMDMD_ini.data(comp_num,:),'Color',[0.7 0.7 0.7],'LineStyle',':','LineWidth',2)
        plot(x_dir,opt_XX.XX_sys(comp_num,:),      'k-','LineWidth',2)
        plot(x_dir,result_POD.XX(comp_num,:),      'g-','LineWidth',2)
        plot(x_dir,result_EMDMD.XX(comp_num,:),    'r-','LineWidth',2)
        xlim([0 n/10]) ;
        ylim([(min(data_EMDMD_ini.data(comp_num,:))+2*min(opt_XX.XX_sys(comp_num,:)))/3 (max(data_EMDMD_ini.data(comp_num,:))+2*max(opt_XX.XX_sys(comp_num,:)))/3]) ;
        xlabel('Time step','color',[0 0 0],'FontSize',12) ;
        ylabel('Value',      'color',[0 0 0],'FontSize',12) ;
        saveas(gcf,['output/plot/specific_data_EMDMD_wo_noise_',data_name,'_',num2str(ense_num),'.fig']);
        saveas(gcf,['output/plot/specific_data_EMDMD_wo_noise_',data_name,'_',num2str(ense_num),'.png']);
        saveas(gcf,['output/plot/specific_data_EMDMD_wo_noise_',data_name,'_',num2str(ense_num),'.emf']);
        saveas(gcf,['output/plot/specific_data_EMDMD_wo_noise_',data_name,'_',num2str(ense_num),'.eps'],'epsc' ) ;
    end

    % Eigenvalue
    figure(1); hold on;
    title('Eigenvalue of EMDMD (red) and DMD (green)')
    box on ;
    set(gcf,'Position',[0,0,360,300]);
    set(gcf,'defaultAxesFontName','Arial');
    set(gcf,'defaultTextFontName','Arial');
    set(gcf,'defaultAxesFontSize',9);
    set(gcf,'defaultTextFontSize',9);
    set(gca,'FontSize',9);
    plot(cos([0:0.01:2*pi]),sin([0:0.01:2*pi]),'k--','LineWidth',1) ;
    scatter(real(DMD_evals),  imag(DMD_evals),  'gd','LineWidth',1) ;
    scatter(real(EMDMD_evals),imag(EMDMD_evals),'ro','LineWidth',1) ;
    scatter(real(true_evals), imag(true_evals), 'ks','LineWidth',1) ;
    xlim([0.7 1.1]) ;
    ylim([0  0.4]) ;
    xlabel('Real part',     'color',[0 0 0],'FontSize',12) ;
    ylabel('Imaginary part','color',[0 0 0],'FontSize',12) ;
    saveas(gcf,['output/plot/eig_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'.fig']) ; 
    saveas(gcf,['output/plot/eig_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'.emf']) ;
    saveas(gcf,['output/plot/eig_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'.png']) ;

    % Reconstruction error
    figure(11); hold on;
    title('Reconstruction error of EMDMD and DMD')
    box on ;
    set(gcf,'Position',[0,0,360,300]);
    set(gcf,'defaultAxesFontName','Arial');
    set(gcf,'defaultTextFontName','Arial');
    set(gcf,'defaultAxesFontSize',9);
    set(gcf,'defaultTextFontSize',9);
    set(gca,'FontSize',9);
    X = categorical({'EMDMD','POD'}) ;
    X = reordercats(X,{'EMDMD','POD'}) ;
    Y = [mean(EMDMD_err,1),mean(POD_err,1)] ;
    h = bar(X,Y) ;
    h.FaceColor = 'flat';
    h.CData(1,:) = [1 0 0] ;
    h.CData(2,:) = [0 1 0] ;
    Y_low  = [std(EMDMD_err,0,1),std(POD_err,0,1)] ;
    Y_high = [std(EMDMD_err,0,1),std(POD_err,0,1)] ;
    er = errorbar(X,Y,Y_low,Y_high) ;
    er.Color = [0,0,0] ;
    er.LineStyle = 'none' ;
    er.LineWidth = 1.2 ;
    ylabel('Reconstruction error','color',[0 0 0],'FontSize',12) ;
    saveas(gcf,['output/plot/rec_err_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'.fig']); 
    saveas(gcf,['output/plot/rec_err_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'.emf']);
    saveas(gcf,['output/plot/rec_err_',data_name,'_mode_',num2str(r),'_Sigma_',num2str(flag_Sigma),'.png']);
end

