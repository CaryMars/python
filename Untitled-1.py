# %%

from gettext import npgettext
from logging import _Level
from os import lseek
from tkinter import CENTER, PROJECTING

# %%
# Import pandas package 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt('Zirnstein_etal_2021_ApJS_MHD_simulation_results.txt',delimiter=" ",unpack=True);

# %%
data = data.transpose()
data_f = pd.DataFrame(data,columns=["radius","theta","phi","den","temp","vx","vy","vz","bx","by","bz","region"])
list(data_f.columns)
data_f.head()

# %%
# data_f[data_f["theta"].map(lambda x:1.570796-0.013090*2<x<1.570796+0.013090*2)]# %%
%matplotlib
# %%
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
test_df1 = data_f[data_f["theta"].map(lambda x: x==1.557706)]

equ_r = test_df1['radius']
equ_theta = test_df1['theta']
equ_phi = test_df1['phi']
equ_x = equ_r * np.sin(equ_theta) * np.cos(equ_phi)
equ_y = equ_r * np.sin(equ_theta) * np.sin(equ_phi)
equ_z = equ_r * np.cos(equ_theta)

equ_den = test_df1['den']

fig = plt.figure()

ax = fig.add_subplot(121)
plt.scatter(equ_phi, equ_r, c=equ_den, marker = "+", cmap = 'hsv')

ax = fig.add_subplot(122, projection = "polar")
plt.scatter(equ_phi, equ_r, c=equ_den, marker = "+", cmap = 'jet')

ax.set_ylim(0,1000)
plt.colorbar(label="Density", orientation="horizontal")

# ax = plt.axes(projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# ax.set_zlim(-1000,1000)

# img = ax.scatter3D(equ_x, equ_y, equ_z, c=equ_den, alpha = 0.02, marker='.')

# fig, ax = plt.subplots()
# ax.scatter(equ_x, equ_y, s=equ_den)
plt.show()
# %%
test_df2 = data_f[data_f["phi"].map(lambda x: (x==3.141593 or x == 0.0))]
# test_df2 = data_f.loc[data_f["phi"].map(lambda x: (x==0.0 | x==3.1415926))]

mer_r = test_df2['radius']
mer_theta = test_df2['theta']
mer_phi = test_df2['phi']
mer_x = mer_r * np.sin(mer_theta) * np.cos(mer_phi)
mer_y = mer_r * np.sin(mer_theta) * np.sin(mer_phi)
mer_z = mer_r * np.cos(mer_theta)

mer_den = test_df2['den']


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
ax.set_zlim(-1000,1000)

img = ax.scatter3D(mer_x, mer_y, mer_z, c=mer_den, alpha = 0.02, marker='.')

# fig, ax = plt.subplots()
# ax.scatter(mer_x, mer_y, s=mer_den)
plt.show()

# %%

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
test_df1 = data_f[data_f["theta"].map(lambda x: x==1.557706)]

equ_r = test_df1['radius']
equ_theta = test_df1['theta']
equ_phi = test_df1['phi']

equ_den = test_df1['den']

fig = plt.figure()

ax = fig.add_subplot(121)
plt.scatter(equ_phi, equ_r, c=equ_den, marker = "+", cmap = 'hsv')

ax = fig.add_subplot(122, projection = "polar")
plt.scatter(equ_phi, equ_r, c=equ_den, marker = "+", cmap = 'jet')

ax.set_ylim(0,1000)
plt.colorbar(label="Density", orientation="horizontal")


test_df2 = data_f[data_f["phi"].map(lambda x: (x==3.141593 or x == 0.0))]

mer_r = test_df2['radius']
mer_theta = test_df2['theta']
mer_phi = test_df2['phi']

mer_den = test_df2['den']

fig = plt.figure()

ax = fig.add_subplot(121)
plt.scatter(mer_theta, mer_r, c=mer_den, marker = "+", cmap = 'hsv')

ax = fig.add_subplot(122, projection = "polar")
plt.scatter(mer_theta, mer_r, c=mer_den, marker = "+", cmap = 'jet')

ax.set_ylim(0,1000)
plt.colorbar(label="Density", orientation="horizontal")

# ax = plt.axes(projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# ax.set_zlim(-1000,1000)

# img = ax.scatter3D(equ_x, equ_y, equ_z, c=equ_den, alpha = 0.02, marker='.')

# fig, ax = plt.subplots()
# ax.scatter(equ_x, equ_y, s=equ_den)
plt.show()
# %%


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
# density
# equ plane
equ_df = data_f[data_f["theta"].map(lambda x: x==1.557706)]
equ_df3 = equ_df[equ_df['region'] == 3]
equ_df2 = equ_df[equ_df['region'] == 2]
equ_df1 = equ_df[equ_df['region'] == 1]
equ_df0 = equ_df[equ_df['region'] == 0]
# equ_df = data_f[data_f["theta"] == 1.557706] & data_f[data_f['region'] == 3]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-y plane")
plt.scatter(equ_df3['radius'] * np.cos(equ_df3['phi']), equ_df3['radius'] * np.sin(equ_df3['phi']), c=equ_df3['den'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="Density", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(equ_df2['radius'] * np.cos(equ_df2['phi']), equ_df2['radius'] * np.sin(equ_df2['phi']), c=equ_df2['den'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="Density", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(equ_df1['radius'] * np.cos(equ_df1['phi']), equ_df1['radius'] * np.sin(equ_df1['phi']), c=equ_df1['den'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="Density", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(equ_df0['radius'] * np.cos(equ_df0['phi']), equ_df0['radius'] * np.sin(equ_df0['phi']), c=equ_df0['den'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="Density", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(equ_df['radius'] * np.cos(equ_df['phi']), equ_df['radius'] * np.sin(equ_df['phi']), c=equ_df['den'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="Density", orientation="horizontal")

plt.savefig('x-y density plot')

# mer plane

mer_df = data_f[data_f["phi"].map(lambda x: (x==3.141593 or x == 0.0))]

mer_df3 = mer_df[mer_df['region'] == 3]
mer_df2 = mer_df[mer_df['region'] == 2]
mer_df1 = mer_df[mer_df['region'] == 1]
mer_df0 = mer_df[mer_df['region'] == 0]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-z plane")
plt.scatter(mer_df3['radius'] * np.sin(mer_df3['theta'] * np.cos(mer_df3['phi'])), mer_df3['radius'] * np.cos(mer_df3['theta']), c=mer_df3['den'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="Density", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(mer_df2['radius'] * np.sin(mer_df2['theta'] * np.cos(mer_df2['phi'])), mer_df2['radius'] * np.cos(mer_df2['theta']), c=mer_df2['den'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="Density", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(mer_df1['radius'] * np.sin(mer_df1['theta'] * np.cos(mer_df1['phi'])), mer_df1['radius'] * np.cos(mer_df1['theta']), c=mer_df1['den'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="Density", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(mer_df0['radius'] * np.sin(mer_df0['theta'] * np.cos(mer_df0['phi'])), mer_df0['radius'] * np.cos(mer_df0['theta']), c=mer_df0['den'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="Density", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(mer_df['radius'] * np.sin(mer_df['theta'] * np.cos(mer_df['phi'])), mer_df['radius'] * np.cos(mer_df['theta']), c=mer_df['den'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="Density", orientation="horizontal")


plt.savefig('x-z density plot')

# equ_r = equ_df['radius']
# equ_theta = equ_df['theta']
# equ_phi = equ_df['phi']

# equ_den = equ_df['den']

# fig = plt.figure(figsize=(12,5.5))

# ax = fig.add_subplot(121)

# plt.scatter(equ_phi, equ_r, c=equ_den, marker = "+", cmap = 'hsv')
# ax.tricontourf(equ_phi, equ_r, equ_den, levels = 256, cmap = 'hsv')

# ax = fig.add_subplot(122)
# ax.tricontourf(equ_r * np.cos(equ_phi), equ_r * np.sin(equ_phi), equ_den, levels = 256, cmap = 'jet')
# plt.scatter(equ_r * np.cos(equ_phi), equ_r * np.sin(equ_phi), c=equ_den, marker = "+", cmap="hsv")

# test_df2 = data_f[data_f["phi"].map(lambda x: (x==3.141593 or x == 0.0))]

# mer_r = test_df2['radius']
# mer_theta = test_df2['theta']
# mer_phi = test_df2['phi']

# mer_den = test_df2['den']

# fig = plt.figure()

# ax = fig.add_subplot(121)
# plt.scatter(mer_theta, mer_r, c=mer_den, marker = "+", cmap = 'hsv')

# ax = fig.add_subplot(122, projection = "polar")
# plt.scatter(mer_theta, mer_r, c=mer_den, marker = "+", cmap = 'jet')

# ax.set_ylim(0,1000)
# plt.colorbar(label="Density", orientation="horizontal")

# ax = plt.axes(projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# ax.set_zlim(-1000,1000)

# img = ax.scatter3D(equ_x, equ_y, equ_z, c=equ_den, alpha = 0.02, marker='.')

# fig, ax = plt.subplots()
# ax.scatter(equ_x, equ_y, s=equ_den)
plt.show()
# %%
test_variable = mer_df.loc[:,'phi']
mer_df.loc['new'] = test_variable
mer_df.to_csv('out.csv')
# %%
 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt('Zirnstein_etal_2021_ApJS_MHD_simulation_results.txt',delimiter=" ",unpack=True);

data = data.transpose()
data_f = pd.DataFrame(data,columns=["radius","theta","phi","den","temp","vx","vy","vz","bx","by","bz","region"])

X = data_f['radius'] * np.sin(data_f['theta']) * np.cos(data_f['phi'])
Y = data_f['radius'] * np.sin(data_f['theta']) * np.sin(data_f['phi'])
Z = data_f['radius'] * np.cos(data_f['theta'])

data_f.loc[:,'X'] = X
data_f.loc[:,'Y'] = Y
data_f.loc[:,'Z'] = Z

data_f.to_csv('out.csv')
# %%
# velocity plot

# vxsity
# equ plane
equ_df = data_f[data_f["theta"].map(lambda x: x==1.557706)]
equ_df3 = equ_df[equ_df['region'] == 3]
equ_df2 = equ_df[equ_df['region'] == 2]
equ_df1 = equ_df[equ_df['region'] == 1]
equ_df0 = equ_df[equ_df['region'] == 0]
# equ_df = data_f[data_f["theta"] == 1.557706] & data_f[data_f['region'] == 3]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-y plane")
plt.scatter(equ_df3['radius'] * np.cos(equ_df3['phi']), equ_df3['radius'] * np.sin(equ_df3['phi']), c=equ_df3['vx'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="velocity x", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(equ_df2['radius'] * np.cos(equ_df2['phi']), equ_df2['radius'] * np.sin(equ_df2['phi']), c=equ_df2['vx'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="velocity x", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(equ_df1['radius'] * np.cos(equ_df1['phi']), equ_df1['radius'] * np.sin(equ_df1['phi']), c=equ_df1['vx'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="velocity x", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(equ_df0['radius'] * np.cos(equ_df0['phi']), equ_df0['radius'] * np.sin(equ_df0['phi']), c=equ_df0['vx'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="velocity x", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(equ_df['radius'] * np.cos(equ_df['phi']), equ_df['radius'] * np.sin(equ_df['phi']), c=equ_df['vx'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="velocity x", orientation="horizontal")

plt.savefig('x-y vx plot')

# mer plane

mer_df = data_f[data_f["phi"].map(lambda x: (x==3.141593 or x == 0.0))]

mer_df3 = mer_df[mer_df['region'] == 3]
mer_df2 = mer_df[mer_df['region'] == 2]
mer_df1 = mer_df[mer_df['region'] == 1]
mer_df0 = mer_df[mer_df['region'] == 0]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-z plane")
plt.scatter(mer_df3['radius'] * np.sin(mer_df3['theta'] * np.cos(mer_df3['phi'])), mer_df3['radius'] * np.cos(mer_df3['theta']), c=mer_df3['vx'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="velocity x", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(mer_df2['radius'] * np.sin(mer_df2['theta'] * np.cos(mer_df2['phi'])), mer_df2['radius'] * np.cos(mer_df2['theta']), c=mer_df2['vx'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="velocity x", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(mer_df1['radius'] * np.sin(mer_df1['theta'] * np.cos(mer_df1['phi'])), mer_df1['radius'] * np.cos(mer_df1['theta']), c=mer_df1['vx'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="velocity x", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(mer_df0['radius'] * np.sin(mer_df0['theta'] * np.cos(mer_df0['phi'])), mer_df0['radius'] * np.cos(mer_df0['theta']), c=mer_df0['vx'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="velocity x", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(mer_df['radius'] * np.sin(mer_df['theta'] * np.cos(mer_df['phi'])), mer_df['radius'] * np.cos(mer_df['theta']), c=mer_df['vx'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="velocity x", orientation="horizontal")


plt.savefig('x-z vx plot')

# %%

# vxsity
# equ plane
equ_df = data_f[data_f["theta"].map(lambda x: x==1.557706)]
equ_df3 = equ_df[equ_df['region'] == 3]
equ_df2 = equ_df[equ_df['region'] == 2]
equ_df1 = equ_df[equ_df['region'] == 1]
equ_df0 = equ_df[equ_df['region'] == 0]
# equ_df = data_f[data_f["theta"] == 1.557706] & data_f[data_f['region'] == 3]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-y plane")
plt.scatter(equ_df3['radius'] * np.cos(equ_df3['phi']), equ_df3['radius'] * np.sin(equ_df3['phi']), c=equ_df3['vy'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="velocity y", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(equ_df2['radius'] * np.cos(equ_df2['phi']), equ_df2['radius'] * np.sin(equ_df2['phi']), c=equ_df2['vy'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="velocity y", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(equ_df1['radius'] * np.cos(equ_df1['phi']), equ_df1['radius'] * np.sin(equ_df1['phi']), c=equ_df1['vy'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="velocity y", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(equ_df0['radius'] * np.cos(equ_df0['phi']), equ_df0['radius'] * np.sin(equ_df0['phi']), c=equ_df0['vy'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="velocity y", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(equ_df['radius'] * np.cos(equ_df['phi']), equ_df['radius'] * np.sin(equ_df['phi']), c=equ_df['vy'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="velocity y", orientation="horizontal")

plt.savefig('x-y vy plot')

# mer plane

mer_df = data_f[data_f["phi"].map(lambda x: (x==3.141593 or x == 0.0))]

mer_df3 = mer_df[mer_df['region'] == 3]
mer_df2 = mer_df[mer_df['region'] == 2]
mer_df1 = mer_df[mer_df['region'] == 1]
mer_df0 = mer_df[mer_df['region'] == 0]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-z plane")
plt.scatter(mer_df3['radius'] * np.sin(mer_df3['theta'] * np.cos(mer_df3['phi'])), mer_df3['radius'] * np.cos(mer_df3['theta']), c=mer_df3['vy'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="velocity y", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(mer_df2['radius'] * np.sin(mer_df2['theta'] * np.cos(mer_df2['phi'])), mer_df2['radius'] * np.cos(mer_df2['theta']), c=mer_df2['vy'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="velocity y", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(mer_df1['radius'] * np.sin(mer_df1['theta'] * np.cos(mer_df1['phi'])), mer_df1['radius'] * np.cos(mer_df1['theta']), c=mer_df1['vy'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="velocity y", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(mer_df0['radius'] * np.sin(mer_df0['theta'] * np.cos(mer_df0['phi'])), mer_df0['radius'] * np.cos(mer_df0['theta']), c=mer_df0['vy'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="velocity y", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(mer_df['radius'] * np.sin(mer_df['theta'] * np.cos(mer_df['phi'])), mer_df['radius'] * np.cos(mer_df['theta']), c=mer_df['vy'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="velocity y", orientation="horizontal")


plt.savefig('x-z vy plot')

# %%

# vxsity
# equ plane
equ_df = data_f[data_f["theta"].map(lambda x: x==1.557706)]
equ_df3 = equ_df[equ_df['region'] == 3]
equ_df2 = equ_df[equ_df['region'] == 2]
equ_df1 = equ_df[equ_df['region'] == 1]
equ_df0 = equ_df[equ_df['region'] == 0]
# equ_df = data_f[data_f["theta"] == 1.557706] & data_f[data_f['region'] == 3]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-y plane")
plt.scatter(equ_df3['radius'] * np.cos(equ_df3['phi']), equ_df3['radius'] * np.sin(equ_df3['phi']), c=equ_df3['vz'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="velocity z", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(equ_df2['radius'] * np.cos(equ_df2['phi']), equ_df2['radius'] * np.sin(equ_df2['phi']), c=equ_df2['vz'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="velocity z", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(equ_df1['radius'] * np.cos(equ_df1['phi']), equ_df1['radius'] * np.sin(equ_df1['phi']), c=equ_df1['vz'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="velocity z", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(equ_df0['radius'] * np.cos(equ_df0['phi']), equ_df0['radius'] * np.sin(equ_df0['phi']), c=equ_df0['vz'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="velocity z", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(equ_df['radius'] * np.cos(equ_df['phi']), equ_df['radius'] * np.sin(equ_df['phi']), c=equ_df['vz'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="velocity z", orientation="horizontal")

plt.savefig('x-y vz plot')

# mer plane

mer_df = data_f[data_f["phi"].map(lambda x: (x==3.141593 or x == 0.0))]

mer_df3 = mer_df[mer_df['region'] == 3]
mer_df2 = mer_df[mer_df['region'] == 2]
mer_df1 = mer_df[mer_df['region'] == 1]
mer_df0 = mer_df[mer_df['region'] == 0]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-z plane")
plt.scatter(mer_df3['radius'] * np.sin(mer_df3['theta'] * np.cos(mer_df3['phi'])), mer_df3['radius'] * np.cos(mer_df3['theta']), c=mer_df3['vz'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="velocity z", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(mer_df2['radius'] * np.sin(mer_df2['theta'] * np.cos(mer_df2['phi'])), mer_df2['radius'] * np.cos(mer_df2['theta']), c=mer_df2['vz'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="velocity z", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(mer_df1['radius'] * np.sin(mer_df1['theta'] * np.cos(mer_df1['phi'])), mer_df1['radius'] * np.cos(mer_df1['theta']), c=mer_df1['vz'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="velocity z", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(mer_df0['radius'] * np.sin(mer_df0['theta'] * np.cos(mer_df0['phi'])), mer_df0['radius'] * np.cos(mer_df0['theta']), c=mer_df0['vz'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="velocity z", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(mer_df['radius'] * np.sin(mer_df['theta'] * np.cos(mer_df['phi'])), mer_df['radius'] * np.cos(mer_df['theta']), c=mer_df['vz'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="velocity z", orientation="horizontal")


plt.savefig('x-z vz plot')

# %%

# vxsity
# equ plane
equ_df = data_f[data_f["theta"].map(lambda x: x==1.557706)]
equ_df3 = equ_df[equ_df['region'] == 3]
equ_df2 = equ_df[equ_df['region'] == 2]
equ_df1 = equ_df[equ_df['region'] == 1]
equ_df0 = equ_df[equ_df['region'] == 0]
# equ_df = data_f[data_f["theta"] == 1.557706] & data_f[data_f['region'] == 3]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-y plane")
plt.scatter(equ_df3['radius'] * np.cos(equ_df3['phi']), equ_df3['radius'] * np.sin(equ_df3['phi']), c=equ_df3['bx'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="magnetic bx", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(equ_df2['radius'] * np.cos(equ_df2['phi']), equ_df2['radius'] * np.sin(equ_df2['phi']), c=equ_df2['bx'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="magnetic bx", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(equ_df1['radius'] * np.cos(equ_df1['phi']), equ_df1['radius'] * np.sin(equ_df1['phi']), c=equ_df1['bx'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="magnetic bx", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(equ_df0['radius'] * np.cos(equ_df0['phi']), equ_df0['radius'] * np.sin(equ_df0['phi']), c=equ_df0['bx'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="magnetic bx", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(equ_df['radius'] * np.cos(equ_df['phi']), equ_df['radius'] * np.sin(equ_df['phi']), c=equ_df['bx'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="magnetic bx", orientation="horizontal")

plt.savefig('x-y bx plot')

# mer plane

mer_df = data_f[data_f["phi"].map(lambda x: (x==3.141593 or x == 0.0))]

mer_df3 = mer_df[mer_df['region'] == 3]
mer_df2 = mer_df[mer_df['region'] == 2]
mer_df1 = mer_df[mer_df['region'] == 1]
mer_df0 = mer_df[mer_df['region'] == 0]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-z plane")
plt.scatter(mer_df3['radius'] * np.sin(mer_df3['theta'] * np.cos(mer_df3['phi'])), mer_df3['radius'] * np.cos(mer_df3['theta']), c=mer_df3['bx'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="magnetic bx", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(mer_df2['radius'] * np.sin(mer_df2['theta'] * np.cos(mer_df2['phi'])), mer_df2['radius'] * np.cos(mer_df2['theta']), c=mer_df2['bx'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="magnetic bx", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(mer_df1['radius'] * np.sin(mer_df1['theta'] * np.cos(mer_df1['phi'])), mer_df1['radius'] * np.cos(mer_df1['theta']), c=mer_df1['bx'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="magnetic bx", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(mer_df0['radius'] * np.sin(mer_df0['theta'] * np.cos(mer_df0['phi'])), mer_df0['radius'] * np.cos(mer_df0['theta']), c=mer_df0['bx'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="magnetic bx", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(mer_df['radius'] * np.sin(mer_df['theta'] * np.cos(mer_df['phi'])), mer_df['radius'] * np.cos(mer_df['theta']), c=mer_df['bx'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="magnetic bx", orientation="horizontal")


plt.savefig('x-z bx plot')

# %%

# vxsity
# equ plane
equ_df = data_f[data_f["theta"].map(lambda x: x==1.557706)]
equ_df3 = equ_df[equ_df['region'] == 3]
equ_df2 = equ_df[equ_df['region'] == 2]
equ_df1 = equ_df[equ_df['region'] == 1]
equ_df0 = equ_df[equ_df['region'] == 0]
# equ_df = data_f[data_f["theta"] == 1.557706] & data_f[data_f['region'] == 3]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-y plane")
plt.scatter(equ_df3['radius'] * np.cos(equ_df3['phi']), equ_df3['radius'] * np.sin(equ_df3['phi']), c=equ_df3['by'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="magnetic by", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(equ_df2['radius'] * np.cos(equ_df2['phi']), equ_df2['radius'] * np.sin(equ_df2['phi']), c=equ_df2['by'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="magnetic by", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(equ_df1['radius'] * np.cos(equ_df1['phi']), equ_df1['radius'] * np.sin(equ_df1['phi']), c=equ_df1['by'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="magnetic by", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(equ_df0['radius'] * np.cos(equ_df0['phi']), equ_df0['radius'] * np.sin(equ_df0['phi']), c=equ_df0['by'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="magnetic by", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(equ_df['radius'] * np.cos(equ_df['phi']), equ_df['radius'] * np.sin(equ_df['phi']), c=equ_df['by'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="magnetic by", orientation="horizontal")

plt.savefig('x-y by plot')

# mer plane

mer_df = data_f[data_f["phi"].map(lambda x: (x==3.141593 or x == 0.0))]

mer_df3 = mer_df[mer_df['region'] == 3]
mer_df2 = mer_df[mer_df['region'] == 2]
mer_df1 = mer_df[mer_df['region'] == 1]
mer_df0 = mer_df[mer_df['region'] == 0]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-z plane")
plt.scatter(mer_df3['radius'] * np.sin(mer_df3['theta'] * np.cos(mer_df3['phi'])), mer_df3['radius'] * np.cos(mer_df3['theta']), c=mer_df3['by'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="magnetic by", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(mer_df2['radius'] * np.sin(mer_df2['theta'] * np.cos(mer_df2['phi'])), mer_df2['radius'] * np.cos(mer_df2['theta']), c=mer_df2['by'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="magnetic by", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(mer_df1['radius'] * np.sin(mer_df1['theta'] * np.cos(mer_df1['phi'])), mer_df1['radius'] * np.cos(mer_df1['theta']), c=mer_df1['by'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="magnetic by", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(mer_df0['radius'] * np.sin(mer_df0['theta'] * np.cos(mer_df0['phi'])), mer_df0['radius'] * np.cos(mer_df0['theta']), c=mer_df0['by'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="magnetic by", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(mer_df['radius'] * np.sin(mer_df['theta'] * np.cos(mer_df['phi'])), mer_df['radius'] * np.cos(mer_df['theta']), c=mer_df['by'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="magnetic by", orientation="horizontal")


plt.savefig('x-z by plot')

# %%

# vxsity
# equ plane
equ_df = data_f[data_f["theta"].map(lambda x: x==1.557706)]
equ_df3 = equ_df[equ_df['region'] == 3]
equ_df2 = equ_df[equ_df['region'] == 2]
equ_df1 = equ_df[equ_df['region'] == 1]
equ_df0 = equ_df[equ_df['region'] == 0]
# equ_df = data_f[data_f["theta"] == 1.557706] & data_f[data_f['region'] == 3]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-y plane")
plt.scatter(equ_df3['radius'] * np.cos(equ_df3['phi']), equ_df3['radius'] * np.sin(equ_df3['phi']), c=equ_df3['bz'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="magnetic bz", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(equ_df2['radius'] * np.cos(equ_df2['phi']), equ_df2['radius'] * np.sin(equ_df2['phi']), c=equ_df2['bz'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="magnetic bz", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(equ_df1['radius'] * np.cos(equ_df1['phi']), equ_df1['radius'] * np.sin(equ_df1['phi']), c=equ_df1['bz'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="magnetic bz", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(equ_df0['radius'] * np.cos(equ_df0['phi']), equ_df0['radius'] * np.sin(equ_df0['phi']), c=equ_df0['bz'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="magnetic bz", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(equ_df['radius'] * np.cos(equ_df['phi']), equ_df['radius'] * np.sin(equ_df['phi']), c=equ_df['bz'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="magnetic bz", orientation="horizontal")

plt.savefig('x-y bz plot')

# mer plane

mer_df = data_f[data_f["phi"].map(lambda x: (x==3.141593 or x == 0.0))]

mer_df3 = mer_df[mer_df['region'] == 3]
mer_df2 = mer_df[mer_df['region'] == 2]
mer_df1 = mer_df[mer_df['region'] == 1]
mer_df0 = mer_df[mer_df['region'] == 0]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-z plane")
plt.scatter(mer_df3['radius'] * np.sin(mer_df3['theta'] * np.cos(mer_df3['phi'])), mer_df3['radius'] * np.cos(mer_df3['theta']), c=mer_df3['bz'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="magnetic bz", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(mer_df2['radius'] * np.sin(mer_df2['theta'] * np.cos(mer_df2['phi'])), mer_df2['radius'] * np.cos(mer_df2['theta']), c=mer_df2['bz'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="magnetic bz", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(mer_df1['radius'] * np.sin(mer_df1['theta'] * np.cos(mer_df1['phi'])), mer_df1['radius'] * np.cos(mer_df1['theta']), c=mer_df1['bz'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="magnetic bz", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(mer_df0['radius'] * np.sin(mer_df0['theta'] * np.cos(mer_df0['phi'])), mer_df0['radius'] * np.cos(mer_df0['theta']), c=mer_df0['bz'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="magnetic bz", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(mer_df['radius'] * np.sin(mer_df['theta'] * np.cos(mer_df['phi'])), mer_df['radius'] * np.cos(mer_df['theta']), c=mer_df['bz'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="magnetic bz", orientation="horizontal")


plt.savefig('x-z bz plot')

# %%

# vxsity
# equ plane
equ_df = data_f[data_f["theta"].map(lambda x: x==1.557706)]
equ_df3 = equ_df[equ_df['region'] == 3]
equ_df2 = equ_df[equ_df['region'] == 2]
equ_df1 = equ_df[equ_df['region'] == 1]
equ_df0 = equ_df[equ_df['region'] == 0]
# equ_df = data_f[data_f["theta"] == 1.557706] & data_f[data_f['region'] == 3]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-y plane")
plt.scatter(equ_df3['radius'] * np.cos(equ_df3['phi']), equ_df3['radius'] * np.sin(equ_df3['phi']), c=equ_df3['temp'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="temp temp", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(equ_df2['radius'] * np.cos(equ_df2['phi']), equ_df2['radius'] * np.sin(equ_df2['phi']), c=equ_df2['temp'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="temp temp", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(equ_df1['radius'] * np.cos(equ_df1['phi']), equ_df1['radius'] * np.sin(equ_df1['phi']), c=equ_df1['temp'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="temp temp", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(equ_df0['radius'] * np.cos(equ_df0['phi']), equ_df0['radius'] * np.sin(equ_df0['phi']), c=equ_df0['temp'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="temp temp", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(equ_df['radius'] * np.cos(equ_df['phi']), equ_df['radius'] * np.sin(equ_df['phi']), c=equ_df['temp'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="temp temp", orientation="horizontal")

plt.savefig('x-y temp plot')

# mer plane

mer_df = data_f[data_f["phi"].map(lambda x: (x==3.141593 or x == 0.0))]

mer_df3 = mer_df[mer_df['region'] == 3]
mer_df2 = mer_df[mer_df['region'] == 2]
mer_df1 = mer_df[mer_df['region'] == 1]
mer_df0 = mer_df[mer_df['region'] == 0]
fig = plt.figure(figsize=(12,16))
#  region 3
ax = fig.add_subplot(221)
plt.title("x-z plane")
plt.scatter(mer_df3['radius'] * np.sin(mer_df3['theta'] * np.cos(mer_df3['phi'])), mer_df3['radius'] * np.cos(mer_df3['theta']), c=mer_df3['temp'], marker = "+", cmap = 'jet')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
plt.colorbar(label="temp temp", orientation="horizontal")
#  region 2
ax = fig.add_subplot(222)
plt.scatter(mer_df2['radius'] * np.sin(mer_df2['theta'] * np.cos(mer_df2['phi'])), mer_df2['radius'] * np.cos(mer_df2['theta']), c=mer_df2['temp'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,250)
ax.set_ylim(-500,500)
plt.colorbar(label="temp temp", orientation="horizontal")
#  region 1
ax = fig.add_subplot(223)
plt.scatter(mer_df1['radius'] * np.sin(mer_df1['theta'] * np.cos(mer_df1['phi'])), mer_df1['radius'] * np.cos(mer_df1['theta']), c=mer_df1['temp'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="temp temp", orientation="horizontal")
# #  region 0
# ax = fig.add_subplot(224)
# plt.scatter(mer_df0['radius'] * np.sin(mer_df0['theta'] * np.cos(mer_df0['phi'])), mer_df0['radius'] * np.cos(mer_df0['theta']), c=mer_df0['temp'], marker = "+", cmap = 'jet')
# ax.set_xlim(-1000,1000)
# ax.set_ylim(-1000,1000)
# plt.colorbar(label="temp temp", orientation="horizontal")
# region all
ax = fig.add_subplot(224)
plt.scatter(mer_df['radius'] * np.sin(mer_df['theta'] * np.cos(mer_df['phi'])), mer_df['radius'] * np.cos(mer_df['theta']), c=mer_df['temp'], marker = "+", cmap = 'jet')
ax.set_xlim(-1000,1000)
ax.set_ylim(-1000,1000)
plt.colorbar(label="temp temp", orientation="horizontal")


plt.savefig('x-z temp plot')

# %%
