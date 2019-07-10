function [ttrX, ttrY, ttrW, ttrVxPhi, ttrVyPhi] = ...
  Quad6D_approx_TTR(gMin, gMax, gN, valueX, valueY, valueW, valueVxPhi, valueVyPhi, tMax, interval)

    global gX gY gW gVxPhi gVyPhi;

    Xdim = 1;
    Ydim = 3;
    Wdim = 6;
    VxPhidims = [2 5];
    VyPhidims = [4 5];

    % Create grid structures for computation
    gX = createGrid(gMin(Xdim), gMax(Xdim), gN(Xdim));
    gY = createGrid(gMin(Ydim), gMax(Ydim), gN(Ydim));
    gW = createGrid(gMin(Wdim), gMax(Wdim), gN(Wdim));
    gVxPhi = createGrid(gMin(VxPhidims), gMax(VxPhidims), gN(VxPhidims), 2);
    gVyPhi = createGrid(gMin(VyPhidims), gMax(VyPhidims), gN(VyPhidims), 2);


    % Time horizon and intermediate results
    tau = 0:interval:tMax;


    ttrX = TD2TTR(gX, valueX, tau);
    ttrY = TD2TTR(gY, valueY, tau);
    ttrW = TD2TTR(gW, valueW, tau);
    ttrVxPhi  = TD2TTR(gVxPhi, valueVxPhi, tau);
    ttrVyPhi  = TD2TTR(gVyPhi, valueVyPhi, tau);

end