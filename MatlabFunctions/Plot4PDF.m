function [] = Plot4PDF(Feature_1d,inds0,inds1,inds2,inds3,title_str,edges,xlims)
    figure
    centers = (edges(1:end-1)+edges(2:end))/2;
    [values] = histcounts(Feature_1d(inds0),edges,'Normalization', 'probability');plot(centers,values,'b');hold on
    [values] = histcounts(Feature_1d(inds1),edges,'Normalization', 'probability');plot(centers,values,'r');hold on
    [values] = histcounts(Feature_1d(inds2),edges,'Normalization', 'probability');plot(centers,values,'g');hold on
    [values] = histcounts(Feature_1d(inds3),edges,'Normalization', 'probability');plot(centers,values,'k');hold on
    if nargin == 8
        xlim(xlims);
    end
    title(title_str);grid on;legend('0','1','2','3');
end