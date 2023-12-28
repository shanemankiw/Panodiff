import numpy as np

# path = '/data/chenziyu/myprojects/ExtremeRotation_code/metadata/my_sun360/dist/sun_360.npy'
path = '/data/chenziyu/myprojects/ExtremeRotation_code/metadata/sun360/test_pair_rotation.npy'
aa = np.load(path, allow_pickle=True).item()


x1 = np.array([ aa[i]['img1']['x'] for i in range(0, len(aa))])
x2 = np.array([ aa[i]['img2']['x'] for i in range(0, len(aa))])

y1 = np.array([ aa[i]['img1']['y'] for i in range(0, len(aa))])
y2 = np.array([ aa[i]['img2']['y'] for i in range(0, len(aa))])

print(x1.max(), x1.min(), x1.mean())
print(x2.max(), x2.min(), x2.mean())
print(y1.max(), y1.min(), y1.mean())
print(y2.max(), y2.min(), y2.mean())