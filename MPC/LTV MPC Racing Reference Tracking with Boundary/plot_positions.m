% Plot history of positions during simulation as well as track
figure
scatter(history_Xi(1,:),history_Xi(2,:))
hold on
axis([track_xmin track_xmax track_ymin track_ymax])
plot(track2.outer(1,:),track2.outer(2,:),'k')
plot(track2.inner(1,:),track2.inner(2,:),'k')
title('Positions')
xlabel('X')
ylabel('Y')
legend('vehicle', 'boundary')