vmax = 0.6;
vmin = -vmax;

figure
subplot(4,1,1)
plot(times(1:length(history_Xi)), history_Xi(1,:))
ylabel("X position")
% xlabel("Time (s)")
grid on
subplot(4,1,2)
plot(times(1:length(history_Xi)), history_Xi(2,:))
ylabel("Y position")
% xlabel("Time (s)")
grid on
subplot(4,1,3)
plot(times(1:length(history_Xi)), history_Xi(3,:))
hold on
%yline(vmax, 'r--');
% yline(vmin, 'r--');
ylabel("Velocity")
% xlabel("Time (s)")
grid on
subplot(4,1,4)
plot(times(1:length(history_Xi)), history_Xi(4,:))
grid on
ylabel("Heading angle (\circ)")
xlabel("Time (s)")