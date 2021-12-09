function cdfdraw(x,Color_Name,Color_Value,Line_Name,Line_Value,Marker_Name,Marker_Value)
num=numel(x);
y=sort(x);
for i=1:num
    percent(i)=i/num;
    value(i)=y(i);
end
H=plot(value,percent,'-^');

ylabel('CDF');
set(H,Color_Name,Color_Value,Line_Name,Line_Value,Marker_Name,Marker_Value)