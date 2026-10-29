from fractions import Fraction as F
a=[F(-1),F(0),F(40,53),F(-37,60),F(1918,3235)]
w=[F(28187,11022267),F(9681557,17031840),F(10794995,327298604),F(86494431,406871840),F(42418984,231244317)]
D=[[0]*5,[0]*5,[F(8060,148877),F(90520,148877),0,0,0],
   [F(-16805621,334527269),F(-54356993,633622128),F(332246,18633899),0,0],
   [F(12147948,222080867),F(61296188,157860699),F(2355019,374043401),F(14402708,628527665),0]]
for i in range(5):
    r1=sum(D[i]); r2=sum(D[i][j]*a[j] for j in range(5))
    print(i, float(r1-(a[i]**2+a[i])/2), float(r2-(a[i]**3-a[i])/6))
print('sum w',float(sum(w)), 'w.a',float(sum(x*y for x,y in zip(w,a))), 'w.a^2',float(sum(x*y*y for x,y in zip(w,a))),'(1/6?)')
for p in range(2,8):
    print(p, float(sum(x*y**p for x,y in zip(w,a))))
