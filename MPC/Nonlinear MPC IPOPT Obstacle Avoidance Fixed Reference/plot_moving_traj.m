% if test_LTV_efficient_osqp
%     scatter(ref(1,i+1:i+N),ref(2,i+1:i+N))
% else
%     scatter(ref(1,i:i+N-1),ref(2,i:i+N-1))
% end
scatter(ref(1,i+1:i+N),ref(2,i+1:i+N))
% scatter(0,0)
hold on
scatter(history_Xi(1,:),history_Xi(2,:))
