import os
import open3d as o3d
import numpy as np
import torch


def transform(pts, trans):
    """
    Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
    Input
        - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0, 2, 1) + trans[:, :3, 3:4]
        return trans_pts.permute(0, 2, 1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T
def to_array(tensor):
    if(not isinstance(tensor, np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def to_o3d_pcd(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(pts))
    return pcd



def estimate_normal(pcd, radius=0.06, max_nn=30):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

def visualize(path):
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False
    raw_src_pcd = o3d.io.read_point_cloud(os.path.join(path, 'src_pcd.ply'))
    raw_tgt_pcd = o3d.io.read_point_cloud(os.path.join(path, 'tgt_pcd.ply'))
    src_kpts = o3d.io.read_point_cloud(os.path.join(path, 'src_kpts.pcd'))
    tgt_kpts = o3d.io.read_point_cloud(os.path.join(path, 'tgt_kpts.pcd'))
    corr = np.loadtxt(os.path.join(path, 'sampled_corr.txt'), dtype=int)
    label = np.loadtxt(os.path.join(path, 'sampled_corr_label.txt'), dtype=int)
    est_trans = np.loadtxt(os.path.join(path, 'est.txt'), dtype=np.float32)
    #gt_trans = np.loadtxt(os.path.join(path, 'GTmat.txt'), dtype=np.float32)
    true_corr = corr[np.where(label > 0)]
    src_kpts.translate((0, 1.25, 0), relative=True)
    tgt_kpts.translate((0, -1.25, 0), relative=True)
    raw_src_pcd.translate((0, 1.25, 0), relative=True)
    raw_tgt_pcd.translate((0, -1.25, 0), relative=True)
    colors1 = []
    colors2 = []
    for l in label:
        if l == 1:
            colors1.append([0, 1, 0])
        else:
            colors1.append([1, 0, 0])
    for i in range(true_corr.shape[0]):
        colors2.append([0, 1, 0])
    if not raw_src_pcd.has_normals():
        estimate_normal(raw_src_pcd)
        estimate_normal(raw_tgt_pcd)
        #estimate_normal(src_kpts)
        #estimate_normal(tgt_kpts)
    raw_src_pcd.paint_uniform_color([1, 0.706, 0])
    raw_tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
    src_kpts.paint_uniform_color([1, 0.706, 0])
    tgt_kpts.paint_uniform_color([0, 0.651, 0.929])
    lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(src_kpts, tgt_kpts, corr)
    lineset.colors = o3d.utility.Vector3dVector(colors1)
    lineset2 = o3d.geometry.LineSet.create_from_point_cloud_correspondences(src_kpts, tgt_kpts, true_corr)
    lineset2.colors = o3d.utility.Vector3dVector(colors2)
    key_to_callback = {ord("K"): change_background_to_black}
    #o3d.visualization.draw_geometries_with_key_callbacks([src_kpts, raw_src_pcd], key_to_callback)
    #o3d.visualization.draw_geometries_with_key_callbacks([tgt_kpts, raw_tgt_pcd], key_to_callback)
    o3d.visualization.draw_geometries_with_key_callbacks([raw_src_pcd, raw_tgt_pcd, src_kpts, tgt_kpts, lineset], key_to_callback)
    o3d.visualization.draw_geometries_with_key_callbacks([raw_src_pcd, raw_tgt_pcd, src_kpts, tgt_kpts, lineset2], key_to_callback)

    src_kpts.translate((0, -1.25, 0), relative=True)
    tgt_kpts.translate((0, 1.25, 0), relative=True)
    raw_src_pcd.translate((0, -1.25, 0), relative=True)
    raw_tgt_pcd.translate((0, 1.25, 0), relative=True)

    raw_src_pcd.transform(est_trans)
    o3d.visualization.draw_geometries_with_key_callbacks([raw_src_pcd, raw_tgt_pcd], key_to_callback)
    return



if __name__ == '__main__':
    visualize('demo/result')
