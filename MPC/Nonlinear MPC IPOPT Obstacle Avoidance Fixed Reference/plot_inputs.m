amax = 0.25;
amin = -amax;
omegamax = 2;
omegamin = -omegamax;

figure
subplot(2,1,1)
stairs(times(1:length(history_U)), history_U(1,:))
hold on
%yline(omegamax, 'r--');
%yline(omegamin, 'r--');
ylabel("Angular Velocity")
% xlabel("Time (s)")
grid on
subplot(2,1,2)
stairs(times(1:length(history_U)), history_U(2,:))
hold on
%yline(amax, 'r--');
%yline(amin, 'r--');
grid on
ylabel("Acceleration")
xlabel("Time (s)")