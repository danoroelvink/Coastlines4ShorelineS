clear all;close all
nmin=30;
lon=ncread("test.nc",'lon');
lat=ncread("test.nc",'lat');
%% 
ok=sum(isnan(lon),2)<nmin;
lon=lon(ok,:);
lat=lat(ok,:);
time=double(ncread("test.nc",'time')+datenum(1984,1,1));
time(1)=nan;
timok=~isnan(time);
time=time(timok);
lon=lon(:,timok);
lat=lat(:,timok);
%% Fill in gaps by interpolation in time
for i=1:size(lon,1)
    loclon=lon(i,:);
    loclat=lat(i,:);
    valid=~isnan(loclon);
    vallon=loclon(valid);
    vallat=loclat(valid);
    valtime=time(valid);
    lon(i,:)=interp1(valtime,vallon,time,"linear","extrap");
    lat(i,:)=interp1(valtime,vallat,time,"linear","extrap");
end
nans=time'*nan;
dist=hypot(diff(lon(:,end)),diff(lat(:,end)));
id=find(dist>0.02);
for j=length(id):-1:1
    lon=[lon(1:id(j),:);nans;lon(id(j)+1:end,:)];
    lat=[lat(1:id(j),:);nans;lat(id(j)+1:end,:)];
end
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
