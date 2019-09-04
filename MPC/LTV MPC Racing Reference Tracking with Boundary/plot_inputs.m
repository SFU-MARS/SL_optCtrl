% Plot history of inputs during simulation
figure
% Plot omega history
subplot(2,1,1)
stairs(times(1:length(times)-1), history_U(1,:))
title('Inputs')
ylabel("Angular Velocity")
grid on
% Plot a history
subplot(2,1,2)
stairs(times(1:length(times)-1), history_U(2,:))
grid on
ylabel("Acceleration")
xlabel("Time (s)")
