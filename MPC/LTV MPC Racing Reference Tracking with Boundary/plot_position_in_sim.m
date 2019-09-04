% Plot solved positions from MPC along horizon
plot(Xi(1,1:size(Xi,2)), Xi(2,1:size(Xi,2)))
hold on 

% Set axis for plot to track edges
axis([track_xmin track_xmax track_ymin track_ymax])

% Plot track boundaries
plot(track2.outer(1,:),track2.outer(2,:),'k')
plot(track2.inner(1,:),track2.inner(2,:),'k')

% Plot references
scatter(ref(1,i+1:i+N),ref(2,i+1:i+N))

% Plot history of positions from t=0 to current t
scatter(history_Xi(1,:),history_Xi(2,:))
hold off
pause(1e-6)
