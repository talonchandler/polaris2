from tqdm import tqdm
from polaris2.geomvis import R3S2toR, R2toR, S2toR, util
from polaris2.micro.micro import det

dip = R3S2toR.xyzj_list([[0,0,1,0,0,0]], shape=[10,10,4],
                        xlabel='10 $\mu$m', title='Single dipole radiator')

d1 = det.FourF()
im1 = d1.dip_to_ebfp(dip)


l2m_2 = S2toR.Jeven([0,0,1,0], title='$Y_{2,-1}$')
l2m_1 = S2toR.Jeven([0,0,0,0,0,1], title='$Y_{2,2}$')

l2m_2.precompute_tripling()
l3 = l2m_2*l2m_1
ell = [l2m_2, l2m_1, l3]

# l2m_2.interact()

for el in ell:
    el.build_actors()

N = 100
for i in tqdm(range(N)):
    util.plot([ell], './out/out'+str(i)+'.png')
    for el in ell:
        el.increment_camera(360/N)
