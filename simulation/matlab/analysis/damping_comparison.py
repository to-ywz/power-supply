import numpy as np,json
from pathlib import Path
L=86e-6; C=7.1047e-4; Vin=155.; Kp=.005; Ki=.1; Ts=1e-5

def plant(G):
 A=np.array([[0,-1/L,Vin/L],[1/C,-G/C,0],[0,0,0.]])
 E=np.eye(3); term=E.copy()
 for n in range(1,45):
  term=term@A*Ts/n;E+=term
 return E[:2,:2],E[:2,2]

def poles(G,Kc):
 Ad,Bd=plant(G)
 # states i,v,q, delayed capacitor current; sampled damping with one-cycle delay
 M=np.zeros((4,4));M[:2,:2]=Ad;M[:2,1]-=Bd*Kp;M[:2,2]=Bd;M[:2,3]=-Bd*Kc
 M[2,1]=-Ki*Ts;M[2,2]=1;M[3,0]=1;M[3,1]=-G
 z=np.linalg.eigvals(M)
 return {'z':[[float(v.real),float(v.imag)] for v in z],'max_abs_z':float(max(abs(z)))}

def transient(G0,G1,Kc):
 Ad,Bd=plant(G1);x=np.array([100*G0,100.]);q=0.;ic_old=0.;data=[]
 for j in range(12000):
  e=100-x[1];u=np.clip(Kp*e+q,-.1,.1)
  d=float(np.clip(100/Vin+u-Kc*ic_old,0,.9))
  data.append([j*Ts,float(x[0]),float(x[1]),d,u])
  ic_new=x[0]-G1*x[1]
  # no internal saturation reached in this diagnostic; simple Euler integrator
  q+=Ki*Ts*e
  x=Ad@x+Bd*d;ic_old=ic_new
 a=np.array(data); err=abs(a[:,2]-100);outside=np.where(err>1)[0]
 return {'IL_min':float(a[:,1].min()),'IL_max':float(a[:,1].max()),'V_min':float(a[:,2].min()),'V_max':float(a[:,2].max()),'D_min':float(a[:,3].min()),'D_max':float(a[:,3].max()),'settle_1V_s':float(a[outside[-1]+1,0]) if len(outside) and outside[-1]+1<len(a) else (0 if not len(outside) else None)}
results=[]
for k in [0.,.001,.002,.003]:
 results.append({'Kc_duty_per_A':k,'poles_G01':poles(.1,k),'poles_G015':poles(.15,k),'load_10_to_15':transient(.1,.15,k),'unload_15_to_10':transient(.15,.1,k)})
Path(__file__).with_name('damping_comparison_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
for r in results:print(r['Kc_duty_per_A'],r['load_10_to_15'],r['unload_15_to_10'])
