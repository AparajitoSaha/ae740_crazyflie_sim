from casadi import MX, vertcat, horzcat, diag, inv_minor, cross, sqrt, cos, sin, mtimes
import numpy as np

class QuadrotorSimplified:
    def __init__(self, mass, arm_length, Ixx, Iyy, Izz, tau, gravity=9.80665):
        self.mass = mass
        self.gravity = gravity
        self.arm_length = arm_length
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.tau = tau
    
    def dynamics(self):
        px = MX.sym('px')
        py = MX.sym('py')
        pz = MX.sym('pz')
        vx = MX.sym('vx')
        vy = MX.sym('vy')
        vz = MX.sym('vz')
        roll = MX.sym('roll')
        pitch = MX.sym('pitch')
        yaw = MX.sym('yaw')
        roll_c = MX.sym('roll_c')
        pitch_c = MX.sym('pitch_c')
        yaw_c = MX.sym('yaw_c')
        thrust = MX.sym('thrust')

        # Setup state and control vectors
        x = vertcat(px,py,pz,vx,vy,vz,roll,pitch,yaw)
        u = vertcat(roll_c,pitch_c,yaw_c,thrust)


        # [TODO] Quadrotor dynamics for position and velocity
        # Instructions: 
        # - This is a simplified model of the Crazyflie drone dynamics, focusing on the 
        #   position and velocity dynamics. The attitude dynamics are assumed to be linear. 
        # - You need to write the dynamics for position and velocity states. 
        # - Note that these are symbolic variable from the Casadi, hence you cannot apply the
        #   usual numpy functions. Instead, you can use the functions that are imported from
        #   the casadi (see line 1). Refer to https://web.casadi.org/docs/#arithmetic-operations
        #   for additional available arithmetic operations permissable for Casadi MX objects.
        # - Be careful about the orientation rotation matrix when calculating accelerations
        #   (vxdot, vydot, vzdot) 
        #
        pxdot = vx
        pydot = vy
        pzdot = vz 

        cr = cos(roll);  sr = sin(roll)
        cp = cos(pitch); sp = sin(pitch)
        cy = cos(yaw);   sy = sin(yaw)
        
        Rx = vertcat(
            horzcat(1,      0,      0),
            horzcat(0,     cr,    -sr),
            horzcat(0,     sr,     cr)
        )
        Ry = vertcat(
            horzcat( cp,    0,   sp),
            horzcat(  0,    1,    0),
            horzcat(-sp,    0,   cp)
        )
        Rz = vertcat(
            horzcat(cy,  -sy, 0),
            horzcat(sy,   cy, 0),
            horzcat( 0,    0, 1)
        )
        R = mtimes(mtimes(Rz, Ry), Rx)          # R = Rz * Ry * Rx
        b3 = mtimes(R, vertcat(0, 0, 1))       # third column
        vxdot = (thrust / self.mass) * b3[0]
        vydot = (thrust / self.mass) * b3[1]
        vzdot = (thrust / self.mass) * b3[2] - self.gravity    

        # Linear first-order attitude dymamics
        rolldot = (roll_c - roll) / self.tau
        pitchdot = (pitch_c - pitch) / self.tau
        yawdot = (yaw_c - yaw) / self.tau
        
        f_expl = vertcat(pxdot,pydot,pzdot,vxdot,vydot,vzdot,rolldot,pitchdot,yawdot)

        return (f_expl, x, u)