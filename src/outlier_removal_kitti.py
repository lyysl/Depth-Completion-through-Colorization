import numpy as np
import tensorflow as tf


def read_one(depth_np, intrinsics_txt):
    
    lidar_one=[]
    intrinsics_matrix=[]

    # img_file = Image.open(velodyne_raw_path+  '/'+i[:27]+'velodyne_raw'+i[32:])
    # depth_png = np.array(img_file, dtype=int)
    # img_file.close()
    depth = depth_np.astype(np.float32) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth,-1)


    # F = open(intrinsics_path+'/'+i[:len(i)-4]+'.txt','r')
    F = open(intrinsics_txt,'r')
    intrinsics_matrix_per=F.readline().split(' ')

    intrinsics_matrix_per=[float(n) for n in intrinsics_matrix_per if not(n=='\n')]
    F.close()
    
    lidar_one.append(depth[:,:,0])
    intrinsics_matrix.append(intrinsics_matrix_per)
        
    return  np.asarray(lidar_one), np.asarray(intrinsics_matrix)



def get_all_points(lidar,intrinsic):

    lidar_32=np.squeeze(lidar).astype(np.float32)
    height,width=np.shape(lidar_32)
    x_axis=[i for i in range(width)]
    x_axis=np.reshape(x_axis,[width,1])
    x_image=np.tile(x_axis, height)
    x_image=np.transpose(x_image)
    y_axis=[i for i in range(height)]
    y_axis=np.reshape(y_axis,[height,1])
    y_image=np.tile(y_axis, width)
    z_image=np.ones((height,width))
    image_coor_tensor=[x_image,y_image,z_image]
    image_coor_tensor=np.asarray(image_coor_tensor).astype(np.float32)
    image_coor_tensor=np.transpose(image_coor_tensor,[1,0,2])

    intrinsic=np.reshape(intrinsic,[3,3]).astype(np.float32)
    intrinsic_inverse=np.linalg.inv(intrinsic)
    points_homo=np.matmul(intrinsic_inverse,image_coor_tensor)

    lidar_32=np.reshape(lidar_32,[height,1,width])
    points_homo=points_homo*lidar_32
    extra_image=np.ones((height,width)).astype(np.float32)
    extra_image=np.reshape(extra_image,[height,1,width])
    points_homo=np.concatenate([points_homo,extra_image],axis=1)

    extrinsic_v_2_c=[[0.007,-1,0,0],[0.0148,0,-1,-0.076],[1,0,0.0148,-0.271],[0,0,0,1]]
    extrinsic_v_2_c=np.reshape(extrinsic_v_2_c,[4,4]).astype(np.float32)
    extrinsic_c_2_v=np.linalg.inv(extrinsic_v_2_c)
    points_lidar=np.matmul(extrinsic_c_2_v,points_homo)


    mask=np.squeeze(lidar)>0.1
    total_points=[points_lidar[:,0,:][mask],points_lidar[:,1,:][mask],points_lidar[:,2,:][mask]]
    total_points=np.asarray(total_points)
    total_points=np.transpose(total_points)
    
    return total_points,x_image[mask],y_image[mask],x_image,y_image

def do_range_projection_try(points,proj_H=64,proj_W=2048,fov_up=3.0,fov_down=-18.0):


    proj_range = np.full((proj_H, proj_W), -1,dtype=np.float32)

    # unprojected range (list of depths for each point)
    unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    proj_xyz = np.full((proj_H, proj_W, 3), -1,dtype=np.float32)

   

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                            dtype=np.int32)

    # for each point, where it is in the range image
    proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
    proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    proj_mask = np.zeros((proj_H, proj_W),dtype=np.int32) 
    
    
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(points, 2, axis=1)


    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]
    return proj_x,proj_y

    
def outlier_removal_mask(lidar,intrinsic, line_num=64,height_offset=100):

    height,width=np.shape(np.squeeze(lidar))

    height_bin=np.round((height-height_offset)/line_num-1)
    width_bin=np.round(width*line_num/(np.sum(lidar>0.1))-1)

        
    total_points,x_indices,y_indices,width_image,height_image=get_all_points(lidar,intrinsic)
    proj_x,proj_y=do_range_projection_try(total_points)

    project_x=np.zeros((height,width))
    project_y=np.zeros((height,width))
    project_x[y_indices,x_indices]=proj_x
    project_y[y_indices,x_indices]=proj_y


    lidar_pre=np.expand_dims(np.squeeze(lidar),axis=0)
    lidar_pre=np.expand_dims(lidar_pre,axis=-1)
    lidar_trunck=tf.image.extract_patches(images=lidar_pre,sizes=[1, height_bin*2+1, width_bin*2+1, 1],strides=[1, 1, 1, 1],rates=[1, 1, 1, 1],padding='SAME')

    
    expand_size=(height_bin*2+1)*(width_bin*2+1)
    expand_size=np.int32(expand_size)
    lidar_pre_expand=tf.tile(lidar_pre.astype(np.float32),[1,1,1,expand_size])
         

    project_x_pre=np.expand_dims(np.squeeze(project_x),axis=0)
    project_x_pre=np.expand_dims(project_x_pre,axis=-1)
    project_x_trunck=tf.image.extract_patches(images=project_x_pre,sizes=[1, height_bin*2+1, width_bin*2+1, 1],strides=[1, 1, 1, 1],rates=[1, 1, 1, 1],padding='SAME')

    project_y_pre=np.expand_dims(np.squeeze(project_y),axis=0)
    project_y_pre=np.expand_dims(project_y_pre,axis=-1)
    project_y_trunck=tf.image.extract_patches(images=project_y_pre,sizes=[1, height_bin*2+1, width_bin*2+1, 1],strides=[1, 1, 1, 1],rates=[1, 1, 1, 1],padding='SAME')

    height_image_pre=np.expand_dims(np.squeeze(height_image).astype(np.double),axis=0)
    height_image_pre=np.expand_dims(height_image_pre,axis=-1)
    height_image_trunck=tf.image.extract_patches(images=height_image_pre,sizes=[1, height_bin*2+1, width_bin*2+1, 1],strides=[1, 1, 1, 1],rates=[1, 1, 1, 1],padding='SAME')

    width_image_pre=np.expand_dims(np.squeeze(width_image).astype(np.double),axis=0)
    width_image_pre=np.expand_dims(width_image_pre,axis=-1)
    width_image_trunck=tf.image.extract_patches(images=width_image_pre,sizes=[1, height_bin*2+1, width_bin*2+1, 1],strides=[1, 1, 1, 1],rates=[1, 1, 1, 1],padding='SAME')


    lidar_residual=lidar_pre-lidar_trunck
        
        
    project_x_residual=project_x_pre-project_x_trunck
    project_y_residual=project_y_pre-project_y_trunck
    height_image_residual=height_image_pre-height_image_trunck
    width_image_residual=width_image_pre-width_image_trunck
    zero_mask=np.logical_and(lidar_pre_expand>0.1,lidar_trunck>0.1)

    x_mask_1=np.logical_and(project_x_residual>0.0000,width_image_residual<=0)
    x_mask_2=np.logical_and(project_x_residual<0.0000,width_image_residual>=0)
    x_mask=np.logical_or(x_mask_1,x_mask_2)
    x_mask=np.logical_and(x_mask,zero_mask)
        
    y_mask_1=np.logical_and(project_y_residual>0,height_image_residual<=0)
    y_mask_2=np.logical_and(project_y_residual<0,height_image_residual>=0)
    y_mask=np.logical_or(y_mask_1,y_mask_2)
    y_mask=np.logical_and(y_mask,zero_mask)
        
    
        
    lidar_mask=np.logical_and(lidar_residual>3.0,lidar_pre>0.01)

    final_mask=np.logical_and(lidar_mask,np.logical_or(x_mask,y_mask))    
    final_mask=np.squeeze(final_mask)
    final_mask=np.sum(final_mask,axis=-1)
    final_mask=np.expand_dims(final_mask>0,axis=0)
    return final_mask

