function [] = Plot4CDF(Feature_1d,inds0,inds1,inds2,inds3,title_str,xlims)
    figure
    [f,x] = ecdf(Feature_1d(inds0));plot(x,f,'b');hold on
    [f,x] = ecdf(Feature_1d(inds1));plot(x,f,'r');hold on
    [f,x] = ecdf(Feature_1d(inds2));plot(x,f,'g');hold on
    [f,x] = ecdf(Feature_1d(inds3));plot(x,f,'k');hold on
    if nargin == 7
        xlim(xlims);
    end
    title(title_str);grid on;legend('0','1','2','3');
end