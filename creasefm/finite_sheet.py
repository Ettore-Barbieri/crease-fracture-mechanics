"""
Paper: E. Barbieri, "Creases in the light of fracture mechanics", Section 7 and Appendix C.

Centre crease in a finite square sheet [-W,W]^2 with free edges (linear Kirchhoff, elastic hinge).
w = w_inf[theta] + w_c.  w_inf: infinite-plate field of the crease (Glauert modes).
w_c: smooth Ritz correction (Legendre products, even in x and y) that makes the edges free:
    a(w_c, v) = -B(w_inf, v)  for all v,
    B(u,v) = -int M_n(u) v_,n ds + int V_n(u) v ds - sum_corners cx*cy*2*M_xy(u)*v   (checked in scripts/checks.py).
Hinge law on the crease: M_yy[w_inf] + M_yy[w_c] = -k (theta - theta_r), Galerkin with weight sin(m tau) sin(tau).
"""
import numpy as np, sympy as sp
from numpy.polynomial import legendre as L
from .glauert import line_stiffness, glauert_coefficients, theta, moment_intensity

_s,_y,_nu=sp.symbols('s y nu')
_K=((1+_nu)*sp.log(_s**2+_y**2)+(1+_nu)+2*(_y**2+_nu*_s**2)/(_s**2+_y**2))/(8*sp.pi)
_DER={}
for p in range(4):
    for q in range(4-p):
        _DER[(p,q)]=sp.lambdify((_s,_y,_nu),sp.diff(_K,_s,p,_y,q) if p+q>0 else _K,'numpy')

def winf_derivs(X,Y,a,nmax,nu,ntau=160):
    """d^{p+q} w_n / dx^p dy^q at points (X,Y) for modes n=1..nmax. Returns dict (p,q)->array (npts, nmax)."""
    tg,tw=np.polynomial.legendre.leggauss(ntau); tau=0.5*np.pi*(tg+1); tw=0.5*np.pi*tw
    t=a*np.cos(tau); jac=a*np.sin(tau)
    n=np.arange(1,nmax+1); S=np.sin(np.outer(tau,n))*(tw*jac)[:,None]      # (ntau,nmax)
    out={}
    for key,f in _DER.items():
        Kv=f(X[:,None]-t[None,:],Y[:,None]*np.ones_like(t)[None,:],nu)     # (npts,ntau)
        out[key]=Kv@S
    return out

class FiniteSheet:
    def __init__(self,W=1.0,D=1.0,nu=0.3,P=24,nb=41,nedge=240,ntau=160):
        self.W,self.D,self.nu,self.P,self.nb,self.ntau=W,D,nu,P,nb,ntau
        self.idx=[(i,j) for i in range(0,P+1,2) for j in range(0,P+1,2) if i+j>=2]
        # stiffness by Gauss quadrature (exact for polynomials)
        g,gw=L.leggauss(P+4); X,Y=np.meshgrid(g*W,g*W,indexing='ij'); WW=np.outer(gw,gw)*W*W
        d=self._basis_derivs(X.ravel(),Y.ravel())
        xx,xy,yy=d['xx'],d['xy'],d['yy']
        w8=WW.ravel()[:,None]
        Kt=(1-nu)*((xx*w8).T@xx+2*(xy*w8).T@xy+(yy*w8).T@yy)+nu*(((xx+yy)*w8).T@(xx+yy))
        self.Kmat=D*Kt
        ge,gwe=L.leggauss(nedge); self.ge=ge*W; self.gwe=gwe*W
    def _leg(self,z,i,der):
        c=np.zeros(i+1); c[i]=1
        if der: c=L.legder(c,der)
        return L.legval(z/self.W,c)/self.W**der
    def _basis_derivs(self,x,y):
        out={k:np.zeros((x.size,len(self.idx))) for k in ['v','x','y','xx','xy','yy']}
        for c,(i,j) in enumerate(self.idx):
            Px=[self._leg(x,i,d) for d in range(3)]; Py=[self._leg(y,j,d) for d in range(3)]
            out['v'][:,c]=Px[0]*Py[0]; out['x'][:,c]=Px[1]*Py[0]; out['y'][:,c]=Px[0]*Py[1]
            out['xx'][:,c]=Px[2]*Py[0]; out['xy'][:,c]=Px[1]*Py[1]; out['yy'][:,c]=Px[0]*Py[2]
        return out
    def boundary_load(self,a):
        """F[j,n] = -B(w_n, v_j) for modes n=1..nb."""
        D,nu,W=self.D,self.nu,self.W; F=np.zeros((len(self.idx),self.nb))
        s=self.ge; w=self.gwe
        for (nx,ny) in [(1,0),(-1,0),(0,1),(0,-1)]:
            if nx: X=np.full_like(s,nx*W); Y=s
            else: X=s; Y=np.full_like(s,ny*W)
            d=winf_derivs(X,Y,a,self.nb,nu,self.ntau)
            dv=self._basis_derivs(X,Y)
            if nx:   # normal x (sign nx), tangent y
                wnn,wss,wns=d[(2,0)],d[(0,2)],nx*d[(1,1)]
                Mn=-D*(wnn+nu*wss); dlap_n=nx*(d[(3,0)]+d[(1,2)])
                dMns_ds=-D*(1-nu)*nx*d[(1,2)]          # d/ds of M_ns along the edge
                vn=nx*dv['x']
            else:
                wnn,wss=d[(0,2)],d[(2,0)]
                Mn=-D*(wnn+nu*wss); dlap_n=ny*(d[(2,1)]+d[(0,3)])
                dMns_ds=-D*(1-nu)*(ny*d[(2,1)])
                vn=ny*dv['y']
            Vn=-D*dlap_n+dMns_ds
            F-= -(vn*w[:,None]).T@Mn + (dv['v']*w[:,None]).T@Vn
        Xc=np.array([W,-W,-W,W]); Yc=np.array([W,W,-W,-W])
        d=winf_derivs(Xc,Yc,a,self.nb,nu,self.ntau); dv=self._basis_derivs(Xc,Yc)
        Mxy=-D*(1-nu)*d[(1,1)]
        F-= -(dv['v']*(Xc*Yc/W**2)[:,None]).T@(2*Mxy)
        return F
    def solve(self,a,k,theta_r,N=400,ntau=400):
        D,nu=self.D,self.nu; Lam=line_stiffness(D,nu)
        n=np.arange(1,N+1)
        # Glauert matrix (as in the module) and RHS
        def c(P):
            P=P.astype(float); out=np.zeros_like(P); ev=(P.astype(int)%2==0); out[ev]=2/(1-P[ev]**2); return out
        Mi,Ni=np.meshgrid(n,n,indexing='ij')
        G=k*0.5*(c(np.abs(Ni-Mi))-c(Ni+Mi)); G[np.diag_indices(N)]+=Lam/(2*a)*n*np.pi/2
        b=np.zeros(N); b[0]=k*theta_r*np.pi/2
        # coupling: w_c = Phi c, c = Kmat^-1 F A
        F=self.boundary_load(a); Cw=np.linalg.solve(self.Kmat,F)            # (nbasis, nb)
        tg,tw=L.leggauss(ntau); tau=0.5*np.pi*(tg+1); tw=0.5*np.pi*tw
        x=a*np.cos(tau); dv=self._basis_derivs(x,np.zeros_like(x))
        Myy_basis=-D*(dv['yy']+nu*dv['xx'])                                  # (ntau,nbasis)
        Wm=np.sin(np.outer(tau,n))*(np.sin(tau)*tw)[:,None]                  # (ntau,N)
        Cpl=np.zeros((N,N)); Cpl[:,:self.nb]=Wm.T@(Myy_basis@Cw)
        A=np.linalg.solve(G+Cpl,b)
        self.last=dict(a=a,k=k,theta_r=theta_r,n=n,A=A,Cw=Cw)
        return n,A
    def w_correction(self,X,Y):
        Cw,A=self.last['Cw'],self.last['A']
        return self._basis_derivs(X,Y)['v']@(Cw@A[:self.nb])
    def energy(self,a,k,theta_r,**kw):
        """Stored energy U (plate + hinge) of the finite sheet, and the Glauert coefficients."""
        n,A=self.solve(a,k,theta_r,**kw)
        return 0.5*k*theta_r*(2*a*theta_r-a*np.pi/2*A[0]), n, A
