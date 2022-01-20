import Metashape
import pickle
import os

doc = Metashape.app.document

pkl_path = r"E:\2022_tanashi_broccoli\01_metashape_projects\bbox.pkl"

if os.path.exists(pkl_path):
    with open(pkl_path, "rb") as handle:
        t, r, c, s = pickle.load(handle)

        T0 = Metashape.Matrix([[t[0], t[1], t[2], t[3]], 
                               [t[4], t[5], t[6], t[7]],
                               [t[8], t[9], t[10], t[11]],
                               [t[12], t[13], t[14], t[15]]])
        R0 = Metashape.Matrix([[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]])
        C0 = Metashape.Vector([c[0], c[1], c[2]])
        S0 = Metashape.Vector([s[0], s[1], s[2]])

else:
    # read chunk 211101
    chunk = doc.chunks[2]
    if chunk.label == "20211101_0":
        T0 = chunk.transform.matrix
        region = chunk.region
        R0 = region.rot
        C0 = region.center
        S0 = region.size
        with open(pkl_path, "wb") as handle:
            pickle.dump([list(T0), list(R0), list(C0), list(S0)], handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(chunk.label)


for chunk in doc.chunks:
    T = chunk.transform.matrix.inv() * T0
    R = Metashape.Matrix([[T[0, 0], T[0, 1], T[0, 2]],[T[1, 0], T[1, 1], T[1, 2]],[T[2, 0], T[2, 1], T[2, 2]]])
    scale = R.row(0).norm()
    R = R * (1 / scale)
    new_region = Metashape.Region()
    new_region.rot = R * R0
    new_region.center = T.mulp(C0)
    new_region.size = S0 * scale / 1.

    chunk.region = new_region