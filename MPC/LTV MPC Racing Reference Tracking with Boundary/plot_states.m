% Plot history of states during simulation
% Plot x history
figure
subplot(4,1,1)
plot(times(1:length(times)), history_Xi(1,:))
title('States')
ylabel("X position")
grid on
% Plot y history
subplot(4,1,2)
plot(times(1:length(times)), history_Xi(2,:))
ylabel("Y position")
grid on
% Plot v history
subplot(4,1,3)
plot(times(1:length(times)), history_Xi(3,:))
ylabel("Velocity")
grid on
% Plot theta history
subplot(4,1,4)
plot(times(1:length(times)), history_Xi(4,:))
grid on
ylabel("Heading angle (\circ)")
xlabel("Time (s)")