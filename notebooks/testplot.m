clear all;close all
lon=ncread("test.nc",'lon');
lat=ncread("test.nc",'lat');
ok=sum(isnan(lon),2)<30;
lon=lon(ok,:);
lat=lat(ok,:);
time=double(ncread("test.nc",'time')+datenum(1984,1,1));
time(1)=nan;
year=datevec(time);
year=year(:,1);
c=colormap("parula")
figure(1)
for i=1:length(time)
    plot(lon(:,i),lat(:,i),'-','Color',c(ceil(i*256/length(time)),:))
    hold on
end
col=colorbar;
set(col,'Ticks',year(2:5:end),'TickLabels',num2str(year(2:5:end)),'TicksMode','auto')
axis equal
