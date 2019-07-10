import matlab.engine
import os
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
#from gym_foo.gym_foo.envs.PlanarQuadEnv_v0 import PlanarQuadEnv_v0

# Note: the state variables and dims could not be equal to that in learning process
# But now, after discussion with Mo, we decide to use the same state variables as learning, but different action variables.
# And we also decide to use system-decomposition

# state variables index
X_IDX = 0
VX_IDX = 1
Y_IDX = 2
VY_IDX = 3
PHI_IDX = 4
W_IDX = 5

# action variables index
T1_IDX = 0
T2_IDX = 1

# goal_phi should not be 0.75 cause that's too hacky.
GOAL_STATE = np.array([4., 0., 4., 0., 0., 0.])


class PlanarQuad_brs_engine(object):
    # Starts and sets up the MATLAB engine that runs in the background.
    def __init__(self):
        self.eng = matlab.engine.start_matlab()

        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.eng.workspace['cur_path'] = cur_path
        self.eng.workspace['home_path'] = os.environ['PROJ_HOME_3']

        self.eng.eval("addpath(genpath([home_path, '/toolboxls']));", nargout=0)
        self.eng.eval("addpath(genpath([home_path, '/helperOC']));", nargout=0)
        self.eng.eval("addpath(genpath(cur_path));", nargout=0)

        self.reset_variables(tMax=10.0)

        if os.path.exists(cur_path + '/value/valueX.mat') and os.path.exists(cur_path + '/value/valueY.mat') and os.path.exists(cur_path + '/value/valueVxPhi.mat') \
            and os.path.exists(cur_path + '/value/valueVyPhi.mat') and os.path.exists(cur_path + '/value/valueW.mat'):
            self.eng.eval("load([cur_path, '/value/valueX.mat']);", nargout=0)
            self.eng.eval("load([cur_path, '/value/valueY.mat']);", nargout=0)
            self.eng.eval("load([cur_path, '/value/valueVxPhi.mat']);", nargout=0)
            self.eng.eval("load([cur_path, '/value/valueVyPhi.mat']);", nargout=0)
            self.eng.eval("load([cur_path, '/value/valueW.mat']);", nargout=0)

            self.valueX = self.eng.workspace['valueX']
            self.valueY = self.eng.workspace['valueY']
            self.valueVxPhi = self.eng.workspace['valueVxPhi']
            self.valueVyPhi = self.eng.workspace['valueVyPhi']
            self.valueW = self.eng.workspace['valueW']
            print("load value function from disk!")
        else:
            print("computing value function from scratch!")
            self.get_init_target()
            self.get_value_function()


        self.value_interpolation()

    def reset_variables(self, tMax=15.0, interval=0.1, nPoints=41):

        self.state_dim = len(GOAL_STATE)
        self.Thrustmin = 0
        self.Thrustmax = 1.0 * 1.25 * 9.81

        # goal center and range
        self.goal_state  = matlab.double([[GOAL_STATE[0]],[GOAL_STATE[1]],[GOAL_STATE[2]],[GOAL_STATE[3]],[GOAL_STATE[4]],[GOAL_STATE[5]]])
        self.goal_radius = matlab.double([[0.5],[0.25],[0.5],[0.25],[np.pi/6],[0.25]])

        # computational range
        self.gMin = matlab.double([[-5.], [-2.], [-5.], [-2.], [-np.pi], [-np.pi/2]])
        self.gMax = matlab.double([[5.], [2.], [5.], [2.], [np.pi], [np.pi/2]])
        self.nPoints = nPoints
        self.gN = matlab.double((self.nPoints * np.ones((self.state_dim, 1))).tolist())
        self.axis_coords = [np.linspace(self.gMin[i][0], self.gMax[i][0], nPoints) for i in range(self.state_dim)]

        # In quadrotor env, target region is set to rectangle, not cylinder
        self.goalRectAndState = matlab.double([[GOAL_STATE[0]-1],[GOAL_STATE[2]-1],[GOAL_STATE[0]+1],[GOAL_STATE[2]+1],
                                               [GOAL_STATE[5]], [GOAL_STATE[1]], [GOAL_STATE[3]], [GOAL_STATE[4]]])


        self.T1Min = self.T2Min = float(self.Thrustmin)
        self.T1Max = self.T2Max = float(self.Thrustmax)
        self.wRange = matlab.double([[-np.pi/2, np.pi/2]])
        self.vxRange = matlab.double([[-2., 2.]])
        self.vyRange = matlab.double([[-2., 2.]])

        self.tMax = float(tMax)
        self.interval = float(interval)

    def get_init_target(self):
        (self.initTargetAreaX, self.initTargetAreaY, self.initTargetAreaW, self.initTargetAreaVxPhi, self.initTargetAreaVyPhi) = \
            self.eng.Quad6D_create_init_target(self.gMin,
                                               self.gMax,
                                               self.gN,
                                               self.goalRectAndState,
                                               self.goal_radius,
                                               nargout=5)
        self.eng.workspace['initTargetAreaX'] = self.initTargetAreaX
        self.eng.workspace['initTargetAreaY'] = self.initTargetAreaY
        self.eng.workspace['initTargetAreaW'] = self.initTargetAreaW
        self.eng.workspace['initTargetAreaVxPhi'] = self.initTargetAreaVxPhi
        self.eng.workspace['initTargetAreaVyPhi'] = self.initTargetAreaVyPhi

        self.eng.eval("save([cur_path, '/target/initTargetAreaX.mat'], 'initTargetAreaX');", nargout=0)
        self.eng.eval("save([cur_path, '/target/initTargetAreaY.mat'], 'initTargetAreaY');", nargout=0)
        self.eng.eval("save([cur_path, '/target/initTargetAreaW.mat'], 'initTargetAreaW');", nargout=0)
        self.eng.eval("save([cur_path, '/target/initTargetAreaVxPhi.mat'], 'initTargetAreaVxPhi');", nargout=0)
        self.eng.eval("save([cur_path, '/target/initTargetAreaVyPhi.mat'], 'initTargetAreaVyPhi');", nargout=0)
        print("initial target created and saved!")

    def get_value_function(self):
        (self.valueX, self.valueY, self.valueW, self.valueVxPhi, self.valueVyPhi) = \
            self.eng.Quad6D_approx_RS(self.gMin,
                                       self.gMax,
                                       self.gN,
                                       self.T1Min,
                                       self.T1Max,
                                       self.T2Min,
                                       self.T2Max,
                                       self.wRange,
                                       self.vxRange,
                                       self.vyRange,
                                       self.initTargetAreaX,
                                       self.initTargetAreaY,
                                       self.initTargetAreaW,
                                       self.initTargetAreaVxPhi,
                                       self.initTargetAreaVyPhi,
                                       self.tMax,
                                       self.interval,
                                       nargout=5)
        self.eng.workspace['valueX'] = self.valueX
        self.eng.workspace['valueY'] = self.valueY
        self.eng.workspace['valueW'] = self.valueW
        self.eng.workspace['valueVxPhi'] = self.valueVxPhi
        self.eng.workspace['valueVyPhi'] = self.valueVyPhi

        self.eng.eval("save([cur_path, '/value/valueX.mat'], 'valueX');", nargout=0)
        self.eng.eval("save([cur_path, '/value/valueY.mat'], 'valueY');", nargout=0)
        self.eng.eval("save([cur_path, '/value/valueW.mat'], 'valueW');", nargout=0)
        self.eng.eval("save([cur_path, '/value/valueVxPhi.mat'], 'valueVxPhi');", nargout=0)
        self.eng.eval("save([cur_path, '/value/valueVyPhi.mat'], 'valueVyPhi');", nargout=0)
        print("value function calculated and saved!")

    def value_interpolation(self):
        np_valueX = np.asarray(self.valueX)[:,-1]
        np_valueY = np.asarray(self.valueY)[:,-1]
        np_valueW = np.asarray(self.valueW)[:,-1]
        np_valueVxPhi = np.asarray(self.valueVxPhi)[:,:,-1]
        np_valueVyPhi = np.asarray(self.valueVyPhi)[:,:,-1]

        # Here we interpolate based on tabular-based discrete value function
        self.vxphi_value_check = RectBivariateSpline(x=self.axis_coords[VX_IDX],
                                            y=self.axis_coords[PHI_IDX],
                                            z=np_valueVxPhi,
                                            kx=1, ky=1)
        self.vyphi_value_check = RectBivariateSpline(x=self.axis_coords[VY_IDX],
                                            y=self.axis_coords[PHI_IDX],
                                            z=np_valueVyPhi,
                                            kx=1, ky=1)
        self.x_value_check = interp1d(x=self.axis_coords[X_IDX], y=np_valueX, fill_value='extrapolate')
        self.y_value_check = interp1d(x=self.axis_coords[Y_IDX], y=np_valueY, fill_value='extrapolate')
        self.w_value_check = interp1d(x=self.axis_coords[W_IDX], y=np_valueW, fill_value='extrapolate', kind='nearest')
        print("value function interpolation done!")

        return 1

    def evaluate_value(self, states):
        states = np.array(states)
        if states.ndim == 1:
            states = np.reshape(states, (1,-1))

        x_value_checker = self.x_value_check(states[:, X_IDX])
        y_value_checker = self.y_value_check(states[:, Y_IDX])
        w_value_checker = self.w_value_check(states[:, W_IDX])
        vxphi_value_checker = self.vxphi_value_check(states[:, VX_IDX], states[:, PHI_IDX], grid=False)
        vyPhi_value_checker = self.vyphi_value_check(states[:, VY_IDX], states[:, PHI_IDX], grid=False)

        assert not np.isnan(x_value_checker)
        assert not np.isnan(y_value_checker)
        assert not np.isnan(w_value_checker)
        assert not np.isnan(vxphi_value_checker)
        assert not np.isnan(vyPhi_value_checker)

        tmp_value = (x_value_checker, y_value_checker, w_value_checker, vxphi_value_checker, vyPhi_value_checker)
        res = np.max(tmp_value, axis=0)

        return res

        # def get_ttr_function(self):
    #     (self.ttrX, self.ttrY, self.ttrW, self.ttrVxPhi, self.ttrVyPhi) = \
    #         self.eng.Quad6D_approx_TTR(self.gMin,
    #                                    self.gMax,
    #                                    self.gN,
    #                                    self.valueX,
    #                                    self.valueY,
    #                                    self.valueW,
    #                                    self.valueVxPhi,
    #                                    self.valueVyPhi,
    #                                    self.tMax,
    #                                    self.interval,
    #                                    nargout=5)
    #
    #     self.eng.workspace['ttrX'] = self.ttrX
    #     self.eng.workspace['ttrY'] = self.ttrY
    #     self.eng.workspace['ttrW'] = self.ttrW
    #     self.eng.workspace['ttrVxPhi'] = self.ttrVxPhi
    #     self.eng.workspace['ttrVyPhi'] = self.ttrVyPhi
    #
    #     self.eng.eval("save([cur_path, '/ttr/ttrX.mat'], 'ttrX');", nargout=0)
    #     self.eng.eval("save([cur_path, '/ttr/ttrY.mat'], 'ttrY');", nargout=0)
    #     self.eng.eval("save([cur_path, '/ttr/ttrW.mat'], 'ttrW');", nargout=0)
    #     self.eng.eval("save([cur_path, '/ttr/ttrVxPhi.mat'], 'ttrVxPhi');", nargout=0)
    #     self.eng.eval("save([cur_path, '/ttr/ttrVyPhi.mat'], 'ttrVyPhi');", nargout=0)
    #
    # def ttr_interpolation(self):
    #     np_tX = np.asarray(self.ttrX)[:, -1]
    #     np_tY = np.asarray(self.ttrY)[:, -1]
    #     np_tW = np.asarray(self.ttrW)[:, -1]
    #     np_tVxPhi = np.asarray(self.ttrVxPhi)
    #     np_tVyPhi = np.asarray(self.ttrVyPhi)
    #
    #     print('np_tX shape is', np_tX.shape, flush=True)
    #     print('np_tY shape is', np_tY.shape, flush=True)
    #     print('np_tW shape is', np_tW.shape, flush=True)
    #     print('np_tVxPhi shape is', np_tVxPhi.shape, flush=True)
    #     print('np_tVyPhi shape is', np_tVyPhi.shape, flush=True)
    #
    #     # Here we interpolate based on discrete ttr_backup function
    #     self.vxphi_ttr_check = RectBivariateSpline(x=self.axis_coords[VX_IDX],
    #                                         y=self.axis_coords[PHI_IDX],
    #                                         z=np_tVxPhi,
    #                                         kx=1, ky=1)
    #     self.vyphi_ttr_check = RectBivariateSpline(x=self.axis_coords[VY_IDX],
    #                                         y=self.axis_coords[PHI_IDX],
    #                                         z=np_tVyPhi,
    #                                         kx=1, ky=1)
    #     self.x_ttr_check = interp1d(x=self.axis_coords[X_IDX], y=np_tX, fill_value='extrapolate')
    #     self.y_ttr_check = interp1d(x=self.axis_coords[Y_IDX], y=np_tY, fill_value='extrapolate')
    #     self.w_ttr_check = interp1d(x=self.axis_coords[W_IDX], y=np_tW, fill_value='extrapolate', kind='nearest')
    #
    # def evaluate_ttr(self, states):
    #
    #     assert not np.isnan(self.vxphi_ttr_check(states[:, VX_IDX], states[:, PHI_IDX], grid=False))
    #     assert not np.isnan(self.vyphi_ttr_check(states[:, VY_IDX], states[:, PHI_IDX], grid=False))
    #     assert not np.isnan(self.x_ttr_check(states[:, X_IDX]))
    #     assert not np.isnan(self.y_ttr_check(states[:, Y_IDX]))
    #     assert not np.isnan(self.w_ttr_check(states[:, W_IDX]))
    #     tmp_ttr = (self.vxphi_ttr_check(states[:, VX_IDX], states[:, PHI_IDX], grid=False),
    #                 self.vyphi_ttr_check(states[:, VY_IDX], states[:, PHI_IDX], grid=False),
    #                 self.x_ttr_check(states[:, X_IDX]),
    #                 self.y_ttr_check(states[:, Y_IDX]),
    #                 self.w_ttr_check(states[:, W_IDX]))
    #     rslt = np.max(tmp_ttr, axis=0)
    #     return rslt[0]

if __name__ == "__main__":
    quad_engine = PlanarQuad_brs_engine()


