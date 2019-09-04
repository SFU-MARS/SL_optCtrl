for i = 1:size(boundaries,2)
    x_left = boundaries(1,i);
    x_right = boundaries(2,i);
    y_bottom = boundaries(3,i);
    y_top = boundaries(4,i);
    fill([x_left,x_right,x_right,x_left],[y_bottom,y_bottom,y_top,y_top],'red');
    hold on
end
grid on