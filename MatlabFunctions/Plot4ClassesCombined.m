function [] = Plot4ClassesCombined(Feature_2d,sampled_inds_mat,title_str,ylims)
    sampled_inds0 = sampled_inds_mat(:,1);sampled_inds1 = sampled_inds_mat(:,2);
    sampled_inds2 = sampled_inds_mat(:,3);sampled_inds3 = sampled_inds_mat(:,4);
    if nargin == 3
        ylims = [-inf inf];
    end   
    figure;
    plot(Feature_2d(sampled_inds0,:).','b');hold on;ylim(ylims)
    plot(Feature_2d(sampled_inds1,:).','r');hold on;ylim(ylims)
    plot(Feature_2d(sampled_inds2,:).','g');hold on;ylim(ylims)
    plot(Feature_2d(sampled_inds3,:).','k');hold on;ylim(ylims)
    title(title_str);grid on;
end