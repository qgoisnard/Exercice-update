import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

class LinearFrame():
    """
    This class implements a model for a linear model of a planar frame.
    It includes tools for assembling the stiffness matrix and load vector, and plotting
    """

    def __init__(self, nodes, elements):
        """
        Define a planar frame structure given an array of nodes and an array of elements.
        
        input: 
         - nodes:  a nnodes x 2 np.array list of point coordinates
         - elements: a nelements x 2 list of integers, given the node numbers defining each element
        """
        self.nodes = nodes
        self.elements = elements
        self.nnodes = self.nodes.shape[0]
        self.nelements = self.elements.shape[0]
        self.ndof = 3*self.nnodes
        self.Ls = np.array([self.L(e) for e in range(self.nelements)])
        self.angles = np.array([self.angle(e) for e in range(self.nelements)])
        # set default values for stiffness and loadings
        self.EI = np.ones(self.nelements)
        self.ES = 1000*np.ones(self.nelements)
        self.f_x = 0.*np.ones(self.nelements)
        self.f_y = 0.*np.ones(self.nelements)
        self.nodal_force_x = 0.*np.ones(self.nnodes)
        self.nodal_force_y = 0.*np.ones(self.nnodes)
        self.nodal_moment = 0.*np.ones(self.nnodes)
        # initialize to 0 the displacement
        self.U = np.zeros(self.ndof)
        print("Frame with")
        print("%i elements"%self.nelements)
        print("%i nodes"%self.nnodes)
        print("%i dofs"%self.ndof)

    def L(self, e):
        """
        Returns the length of the element e
        """
        dx = self.nodes[self.elements[e]][1]-self.nodes[self.elements[e]][0]
        return np.sqrt(dx[0]**2+dx[1]**2)

    def angle(self, e):
        """
        Returns the orientation angle of the element e
        """
        coord = self.nodes[self.elements[e]]
        delta = coord[1]-coord[0]
        angle = np.arctan2(delta[1],delta[0])
        return angle
    
    def rotation_matrix(self, alpha):
        """
        Transformation matrix to pass from local to global coordinates
        input: the orientation angle 
        output: the 6x6 rotation matrix
        """
        cc = sp.cos(alpha)
        ss = sp.sin(alpha)
        R = sp.Matrix([[cc,ss,0,0,0,0],
               [-ss,cc,0,0,0,0],
               [0,0,1,0,0,0],
               [0,0,0,cc,ss,0],
               [0,0,0,-ss,cc,0],
               [0,0,0,0,0,1]
               ])
        return R

    def dof_map_e(self, e):
        """
        Returns the list of global dofs for the element e
        """
        return np.concatenate((self.elements[e,0]*3+np.arange(3),self.elements[e,1]*3+np.arange(3)))

    def dof_map(self, e, i_local):
        """
        Returns the global dof corresponding to the local dof i_local in the element e
        """
        return self.dof_map_e(e)[i_local]

    def S(self):
        """
        Returns the shape function matrix in sympy format, s being the curbilinear coordinate in the element of lenght L
        """
        L = sp.Symbol('L')
        s = sp.Symbol('s')
        S = sp.Matrix([[1 - s/L, 0, 0, s/L, 0, 0],
                       [0, 1 - 3*s**2/L**2 + 2*s**3/L**3, s - 2*s**2/L + s**3/L**2, 0, 
                        3*s**2/L**2 - 2*s**3/L**3, -s**2/L + s**3/L**2]])
        return S

    def Sn(self, s, e):
        """
        Returns the numerical numpy array of shape functions evaluated at the point s in the element e
        """
        L = self.Ls[e]
        S = np.array(self.S().subs({'L':L, 's':s}))
        return S
    
    def U_e_local_coord(self,e):
        """
        Returns the displacement of the point s in the element e, in local coordinates 
        """
        # Extract the local displacement vector in global coordinates
        U_e = sp.Matrix(self.U[self.dof_map_e(e)])
        # Convert the result in local coordinates
        R = self.rotation_matrix(self.angles[e])  
        U_e_local_coord = R*U_e
        return R*U_e
        
    def uv_local_coord(self,e,s):
        """
        Returns the displacement of the point s in the element e, in local coordinates 
        """
        u_local = self.S()*self.U_e_local_coord(e)
        return u_local.subs({"L": self.Ls[e],"s": s})
     
    def uv_global_coord(self,e,s):
        """
        Returns the displacement of the point s in the element e, in global coordinates 
        """
        R = self.rotation_matrix(self.angles[e]) 
        Rt_red =(R.transpose())[0:2,0:2]
        u_global = Rt_red*self.uv_local_coord(e,s)
        return u_global
        
    def xy(self,e,s):
        """
        Returns the current position of the point s in the element e 
        """
        element = self.elements[e]
        x0coords = self.nodes[element][:,0]
        y0coords = self.nodes[element][:,1]
        x0 = x0coords[0]*(1-s/self.Ls[e])+x0coords[1]*s/self.Ls[e]
        y0 = y0coords[0]*(1-s/self.Ls[e])+y0coords[1]*s/self.Ls[e]
        [u,v] = self.uv_global_coord(e,s)
        return [x0+u, y0+v]

    def plot(self):
        #import matplotlib.plt as plt
        fig, ax = plt.subplots()
        shift = self.nodes.max()*.005
        for (e_num,e) in enumerate(self.elements):
            xcoords = self.nodes[e][:,0]
            ycoords = self.nodes[e][:,1]
            ax.plot(xcoords, ycoords,'o-',lw=2, color='black', ms=10)
        ax.set_xlim(self.nodes[:,0].min()-10*shift, self.nodes[:,0].max()+10*shift)
        ax.set_ylim(self.nodes[:,1].min()-10*shift, self.nodes[:,1].max()+10*shift)
        return ax

    def plot_with_label(self):
        fig, ax = plt.subplots()
        shift = self.nodes.max()*.005
        for (e_num,e) in enumerate(self.elements):
            xcoords = self.nodes[e][:,0]
            ycoords = self.nodes[e][:,1]
            ax.plot(xcoords, ycoords,'o-',lw=2, color='black', ms=10)
            ax.text((xcoords[0]+xcoords[1])/2.+shift, 
                    (ycoords[0]+ycoords[1])/2.+shift, str(e_num), bbox=dict(facecolor='yellow'))
            for i in range(e.size):
                ax.text(xcoords[i]+shift, ycoords[i]+shift, e[i], 
                        verticalalignment = 'bottom', horizontalalignment = 'left')
        ax.set_xlim(self.nodes[:,0].min()-10*shift, self.nodes[:,0].max()+10*shift)
        ax.set_ylim(self.nodes[:,1].min()-10*shift, self.nodes[:,1].max()+10*shift)
        ax.set_aspect('equal', 'datalim')
        return ax

    def plot_displaced(self):
        """
        Plots the displaced configuration of the structure
        """
        for e in range(self.nelements):
            s = sp.Symbol('s')
            xy = self.xy(e,s)
            svalues = np.linspace(0,self.Ls[e],30)
            plt.plot([xy[0].subs({s: sv}) for sv in svalues],[xy[1].subs({s: sv}) for sv in svalues],'b-')
            plt.plot([xy[0].subs({s: 0}),xy[0].subs({s: self.Ls[e]})],
                     [xy[1].subs({s: 0}),xy[1].subs({s: self.Ls[e]})],'ro', ms=7)
            
    def B(self):
        """
        Returns the B 2 x 6 matrix giving the relation between the local displacement vector 
        and the deformations (extension and curvature) 
        """
        L = sp.Symbol('L')
        s = sp.Symbol('s')
        B = sp.Matrix([[-1/L, 0, 0, 1/L, 0, 0], 
                       [0, 2*(-3 + 6*s/L)/L**2, 2*(-2*L + 3*s)/L**2, 0, 2*(3 - 6*s/L)/L**2, 2*(-L + 3*s)/L**2]])
        return B

    def K_local(self):
        """
        Analytical expression of the local stiffness matrix in the local coordinate system
        """
        ES, EI, L = sp.symbols('ES, EI, L')
        s = sp.Symbol('s')
        B = self.B()
        Ke_ext = ES*sp.integrate(B[0,:].transpose()*B[0,:],(s,0,L))
        Ke_bend = EI*sp.integrate(B[1,:].transpose()*B[1,:],(s,0,L))
        Ke = Ke_ext + Ke_bend
        return Ke

    def K_local_rotated(self):
        """
        Gives the analytical expression the local stiffness matrix in the global coordinate system 
        as a function of the orientation angle alpha
        """
        alpha = sp.Symbol("alpha")
        R = self.rotation_matrix(alpha)
        Ke = R.transpose()*self.K_local()*R
        return Ke

    def F_local(self):
        """
        Analytical expression of the local force vector in the local coordinate system 
        as a function of distributed load (f_u, f_v) in the local coordinate system
        and the orientation angle alpha
        """
        f_t, f_n, L = sp.symbols('f_t f_n L')
        s = sp.Symbol('s')
        S = self.S()
        Fe = sp.Matrix([sp.integrate(f_t*S[0,i]+f_n*S[1,i],(s,0,L)) for i in range(6)])
        return Fe
    
    def F_local_rotated(self):
        """
        Analytical expression of the local force vector in the global coordinate system 
        as a function of distributed load (f_x, f_y) in the global coordinate system
        and the orientation angle alpha
        """
        # Get analytical expression in terms of input distributed force vector in local coordinates
        Fe = self.F_local()
        # Express input distributed force in terms of global coordinates 
        f_x, f_y, alpha = sp.symbols('f_x f_y alpha')
        cc = sp.cos(alpha)
        ss = sp.sin(alpha)
        f_t = cc*f_x + ss*f_y
        f_n = -ss*f_x + cc*f_y
        Fe = Fe.subs({"f_t": f_t, "f_n": f_n})
        # Rotate to give force vector in global coordinates
        R = self.rotation_matrix(alpha)
        Fe = R.transpose()*Fe
        return Fe

    def assemble_K(self):
        """
        Returns the global stiffness matrix
        """
        Ke = self.K_local_rotated()
        K = np.zeros([self.ndof,self.ndof])
        # add rotation
        for e in range(self.nelements):
            Ken = np.array(Ke.subs({'EI': self.EI[e], 'ES': self.ES[e], 'L': self.Ls[e], 'alpha': self.angles[e]}))
            Re = np.array(self.rotation_matrix(self.angles[e]))
            for i_local in range(6):
                for j_local in range(6):
                    K[self.dof_map(e, i_local),self.dof_map(e, j_local)] += Ken[i_local,j_local]
        return K

    def assemble_F(self):
        """
        Returns the global force vector
        """
        Fe = self.F_local_rotated()
        F = np.zeros([self.ndof])
        for e in range(self.nelements):
            Fen = np.array(Fe.subs({'f_x': self.f_x[e], 'f_y': self.f_y[e], 'L': self.Ls[e], 'alpha': self.angles[e]}))
            for i_local in range(6):
                F[self.dof_map(e, i_local)] += Fen[i_local]
        return F

    def bc_apply(self, K, F, blocked_dof, bc_values):
        """
        Apply Dirichlet bcs 
        Input 
         - the global stiffness K before applying bcs
         - the global force vector F before applying bcs
         - a list of blocked dofs
         - a list of imposed values for each of the blocked dofs
        """
        for (i, dof) in enumerate(blocked_dof):
            Kbc = K
            Fbc = F
            Kbc[dof, :] = 0
            Kbc[:, dof] = 0
            Kbc[dof, dof] = 1
            Fbc +=  - K[:,dof]*bc_values[i]
            Fbc[dof] = bc_values[i]
        return Kbc, Fbc

    def set_stiffness(self, EI, ES):
        """
        Set the global stiffness of the frame
        Input:
        - EI: an array of length nelements giving the bending stiffness of each element
        - ES: an array of length nelements giving the extensional stiffness of each element

        """
        # Should check that they have the same length of elements
        self.EI = np.array(EI)
        self.ES = np.array(ES)

    def set_distributed_loads(self, f_x, f_y):
        """
        Set the distributed loads, supposed constant in each elements
        Input: 
         - f_x: an array of length nelements giving the x component (global coordinate) 
                of the distributed load in each element
         - f_y: an array of length nelements giving the y component (global coordinate) 
                of the distributed load in each element

        """
        # Should check that it has the same length of elements
        self.f_x = np.array(f_x)
        self.f_y = np.array(f_y)

    def set_point_loads(self, P_x, P_y, C):
        """
        Set the concentrated loads, supposed to be applied at nodes
        Input: 
         - P_x: an array of length nnodes giving the x component (global coordinate) of the load at each node
         - P_y: an array of length nnodes giving the y component (global coordinate) of the load at each node
         - C: an array of length nnodes giving the applied external moment at each node

        """
        # Should check that they have the same length of nodes
        self.nodal_force_x = P_x
        self.nodal_force_y = P_y
        self.nodal_moment = C
        
    def set_displacement(self, U):
        """
        Set a global displacement vector (used for postprocessing)
        """
        # Should check that it has the same length of nodes
        self.U = U