import os,PySide2
dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
import numpy as np
import pylab as pl
from  mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate.rbf import Rbf

# 3 维数据点
x,y=np.mgrid[-np.pi/2:np.pi/2:5j, -np.pi/2:np.pi/2:5j]
z = np.cos(np.sqrt(x**2+y**2))
fig = pl.figure(figsize=(12,6))
ax = fig.gca(projection="3d")
ax.scatter(x,y,z)

# 3 维RBF插值
zz = Rbf(x,y,z)
xx, yy = np.mgrid[-np.pi/2:np.pi/2:50j, -np.pi/2:np.pi/2:50j]
fig = pl.figure(figsize=(12,6))
ax = fig.gca(projection="3d")
ax.plot_surface(xx,yy,zz(xx,yy),rstride=1,cstride=1,cmap=pl.cm.jet)

pl.show()