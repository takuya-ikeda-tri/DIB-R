import cv2
import numpy as np

# Ref: https://qiita.com/koshian2/items/c133e2e10c261b8646bf
# Ref: https://note.nkmk.me/python-opencv-image-warping/


def loadobjtex(meshfile):
    v = []
    vt = []
    f = []
    ft = []
    meshfp = open(meshfile, 'r')
    for line in meshfp.readlines():
        data = line.strip().split(' ')
        data = [da for da in data if len(da) > 0]
        if len(data) != 4 and len(data) != 7 and len(data) != 3:
            continue
        if data[0] == 'v':
            v.append([float(d) for d in data[1:4]])
        if data[0] == 'vt':
            vt.append([float(d) for d in data[1:3]])
        if data[0] == 'f':
            data = [da.split('/') for da in data]
            f.append([int(d[0]) for d in data[1:]])
            ft.append([int(d[1]) for d in data[1:]])
    meshfp.close()

    # torch need int64
    facenp_fx3 = np.array(f, dtype=np.int64) - 1
    ftnp_fx3 = np.array(ft, dtype=np.int64) - 1
    pointnp_px3 = np.array(v, dtype=np.float32)
    uvs = np.array(vt, dtype=np.float32)[:, :2]
    return pointnp_px3, facenp_fx3, ftnp_fx3, uvs


# symmetric over x axis
def get_spherical_coords_x(X):
    import pdb
    pdb.set_trace()
    # X is N x 3
    rad = np.linalg.norm(X, axis=1)
    theta = np.arccos(X[:, 0] / rad)
    # Azimuth
    phi = np.arctan2(X[:, 2], X[:, 1])
    # Normalize both to be between [-1, 1]
    uu = (theta / np.pi) * 2 - 1
    vv = ((phi + np.pi) / (2 * np.pi)) * 2 - 1
    # Return N x 2
    return np.stack([uu, vv], 1)


def savemesh(pointnp_px3, facenp_fx3, fname, partinfo=None):

    if partinfo is None:
        fid = open(fname, 'w')
        for pidx, p in enumerate(pointnp_px3):
            pp = p
            fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))
        for f in facenp_fx3:
            f1 = f + 1
            fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
        fid.close()
    else:
        fid = open(fname, 'w')
        for pidx, p in enumerate(pointnp_px3):
            if partinfo[pidx, -1] == 0:
                pp = p
                color = [1, 0, 0]
            else:
                pp = p
                color = [0, 0, 1]
            fid.write('v %f %f %f %f %f %f\n' %
                      (pp[0], pp[1], pp[2], color[0], color[1], color[2]))
        for f in facenp_fx3:
            f1 = f + 1
            fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
        fid.close()
    return


def savemeshcolor(pointnp_px3, facenp_fx3, fname, color_px3=None):

    if color_px3 is None:
        fid = open(fname, 'w')
        for pidx, p in enumerate(pointnp_px3):
            pp = p
            fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))
        for f in facenp_fx3:
            f1 = f + 1
            fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
        fid.close()
    else:
        fid = open(fname, 'w')
        for pidx, p in enumerate(pointnp_px3):
            pp = p
            color = color_px3[pidx]
            fid.write('v %f %f %f %f %f %f\n' %
                      (pp[0], pp[1], pp[2], color[0], color[1], color[2]))
        for f in facenp_fx3:
            f1 = f + 1
            fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
        fid.close()
    return


def p2f(points_px3, faces_fx3):

    ##########################################################
    # 1 points
    pf0_bxfx3 = points_px3[faces_fx3[:, 0], :]
    pf1_bxfx3 = points_px3[faces_fx3[:, 1], :]
    pf2_bxfx3 = points_px3[faces_fx3[:, 2], :]
    # import ipdb
    # ipdb.set_trace()
    points3d_bxfx9 = np.concatenate((pf0_bxfx3, pf1_bxfx3, pf2_bxfx3), axis=-1)

    return points3d_bxfx9


def savemesh_facecolor(pointnp_px3, facenp_fx3, fname, color_px9=None):
    pointnp_px9 = p2f(pointnp_px3, facenp_fx3)
    # import ipdb
    # ipdb.set_trace()

    fid = open(fname, 'w')
    for pidx, p in enumerate(pointnp_px9):
        pp = p
        color = color_px9[pidx]
        for i_point in range(3):
            fid.write('v %f %f %f %f %f %f\n' %
                      (pp[i_point * 3 + 0], pp[i_point * 3 + 1],
                       pp[i_point * 3 + 2], color[i_point * 3],
                       color[i_point * 3 + 1], color[i_point * 3 + 2]))
    for f_idx in range(pointnp_px9.shape[0]):
        fid.write('f %d %d %d\n' %
                  (f_idx * 3 + 1, f_idx * 3 + 2, f_idx * 3 + 3))

    fid.close()


def crop_triangle(img, dst, three_pts):
    mask = np.zeros_like(dst, dtype=np.float32)
    cv2.fillConvexPoly(mask, three_pts.astype(np.int), (1.0, 1.0, 1.0),
                       cv2.LINE_AA)
    mask = mask.astype(np.uint8)
    img_merged = img * mask + dst * (1 - mask)
    return img_merged


def point_to_spherical_coordinate(point):
    pass
    spherical_coord = [[u1, v1], [u2, v2], [u3, v3]]
    return spherical_coord


if __name__ == "__main__":
    texturefile = 'texture.png'
    meshfile = 'mesh.obj'
    texture = cv2.imread(texturefile)
    # v, f, ft, vt
    pointnp_px3, facenp_fx3, ftnp_fx3, uvs = loadobjtex(meshfile)

    pre_coords = []
    for i, ft in enumerate(ftnp_fx3):
        uv = ft_to_uv(ft, uvs)
        pre_coords.append(uv)

    spherical_coords = []
    for i, face3 in enumerate(facenp_fx3):
        point3 = face_to_point(face3, pointnp_px3)
        spherical_coords.append(point_to_spherical_coordinate(point3))

if __name__ == "__main__":
    im = cv2.imread('lena.png')
    cv2.imshow('test', im)
    back_im = cv2.imread('blue.png')
    back_im = cv2.resize(back_im, im.shape[:2])
    pre_pts = np.array([[0, 0], [256, 511], [511, 0]])
    post_pts = np.array([[0, 511], [256, 256], [0, 256]])
    af = cv2.getAffineTransform(pre_pts.astype(np.float32),
                                post_pts.astype(np.float32))
    converted = cv2.warpAffine(im, af, im.shape[:2])
    # res_im = crop_triangle(im, back_im, post_pts)
    res_im = crop_triangle(converted, back_im, post_pts)
    cv2.imshow('test', res_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # import pdb
    # pdb.set_trace()

    # af = cv2.getAffineTransform(src, dest)
    # converted = cv2.warpAffine(image, af, (size_x, size_y))