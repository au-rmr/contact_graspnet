import os
import sys
import argparse
import numpy as np
import time
import glob
from math import cos, sin, radians
import cv2
import rospy
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose, PoseArray 
import ros_numpy
from threading import RLock
import matplotlib.pyplot as plt
from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image
from aurmr_perception.srv import (
    CaptureObject,
    CaptureObjectRequest,
    ResetBin,
    ResetBinRequest
)
from aurmr_perception.srv import DetectGraspPoses

# from aurmr_tasks.common.tahoma import Tahoma
from tf.transformations import quaternion_from_matrix, translation_from_matrix, euler_matrix, euler_from_matrix, quaternion_from_euler

CAMERA_ROT=-90
SAVE_FILE = "image_test1.npy"
class ContactGraspnetServer():
    def __init__(self) -> None:
        default_dir = "/home/aurmr/workspaces/jack_ws/contact_graspnet/"
        checkpoint_dir = default_dir+'checkpoints/scene_test_2048_bs3_hor_sigma_001'
        global_config = config_utils.load_config(checkpoint_dir, batch_size=1, arg_configs=[])

        # Build the model
        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        # Load weights
        self.grasp_estimator.load_weights(self.sess, saver, checkpoint_dir, mode='test')
        
        self.get_mask_srv = rospy.ServiceProxy('/aurmr_perception/capture_object', CaptureObject)
        self.reset_bin_srv = rospy.ServiceProxy('/aurmr_perception/reset_bin', ResetBin)
        self.detect_grasps_srv = rospy.Service('/grasp_detection/detect_grasps', DetectGraspPoses, self.inference_ros_callback)

        self.get_mask_srv.wait_for_service(timeout=5)
        self.reset_bin_srv.wait_for_service(timeout=5)
        camera_depth_subscriber = message_filters.Subscriber(f'/camera_lower_right/aligned_depth_to_color/image_raw', Image)
        camera_rgb_subscriber = message_filters.Subscriber(f'/camera_lower_right/color/image_raw', Image)
        camera_info_subscriber = message_filters.Subscriber(f'/camera_lower_right/color/camera_info', CameraInfo)
        self.pose_pub = rospy.Publisher("~grasp_pose", PoseStamped)
        self.posearr_pub = rospy.Publisher("~grasp_posearr", PoseArray)
        self.lock = RLock()
        self.depth = None
        self.rgb = None
        self.info = None
        camera_synchronizer = message_filters.TimeSynchronizer([
            camera_depth_subscriber, camera_rgb_subscriber, camera_info_subscriber], 10)
        camera_synchronizer.registerCallback(self.camera_callback)
        
        print("Waiting for images")
        while not rospy.is_shutdown() and self.depth is None:
            rospy.sleep(.1)
        print("Got images")

    def camera_callback(self, depth_msg: Image, rgb_msg: Image, info_msg: CameraInfo):
        with self.lock:
            self.depth = ros_numpy.numpify(depth_msg)
            self.rgb = ros_numpy.numpify(rgb_msg)
            self.info = info_msg

    def inference_ros_callback(self, request):
        return self.inference(ros_numpy.numpify(request.mask))

    def inference_from_ros_manual(self):
        BIN = "3f"
        OBJECT = "bottle"
        reset_bin_req = ResetBinRequest(bin_id=BIN)
        reset_response = self.reset_bin_srv(reset_bin_req)
        print(reset_response)

        capture_obj_req = CaptureObjectRequest(
            bin_id=BIN,
            object_id=OBJECT,
        )
        rospy.sleep(5)
        res = self.get_mask_srv(capture_obj_req)
        status = res.success    
        message = res.message
        segmap = ros_numpy.numpify(res.mask)
        with self.lock:
            np.save(SAVE_FILE, (self.rgb, self.depth, self.info, segmap))
            self.inference(segmap)

    def inference_from_file(self):
        saved = np.load(SAVE_FILE if len(sys.argv) <= 2 else sys.argv[2], allow_pickle=True)
        with self.lock:
            (self.rgb, self.depth, self.info, segmap) = saved
            try:
                segmap = ros_numpy.numpify(segmap)
            except:
                pass
            return self.inference(segmap)

    #Helper function to undo the image rotation that was applied for input to contact graspnet, and to comply with ROS coordinate frame standards
    def transform_grasp(self, grasp):
        theta = radians(CAMERA_ROT)
        undo_rot_matrix = np.asarray([[cos(theta), -sin(theta), 0, 0], [sin(theta), cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        theta = radians(90)
        ros_rot_matrix = np.asarray([[cos(theta), -sin(theta), 0, 0], [sin(theta), cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # undo_rot_matrix = euler_matrix(0, 0, radians(-CAMERA_ROT))
        final_grasp = undo_rot_matrix@grasp@ros_rot_matrix
        # # undo_rot_matrix = euler_matrix(0, 0, radians(-CAMERA_ROT))
        # final_grasp = undo_rot_matrix@grasp
        trans = translation_from_matrix(final_grasp)
        quat = quaternion_from_matrix(final_grasp)
        # euler = list(euler_from_matrix(grasp))
        # # #Rotate orientation to match ROS Coordinate Fram Standards (z up, x forward, y left)
        # # # euler[0] = euler[0] + radians(-90)
        # # # euler[1] = euler[1] + radians(-90)
        # euler[2] = euler[2] + radians(-90)
        # quat = quaternion_from_euler(euler[0], euler[1], euler[2])
        # trans = translation_from_matrix(grasp)
        # quat = quaternion_from_matrix(grasp)

        return trans, quat

    def inference(self, segmap):
        local_regions = True
        filter_grasps=True
        segmap_id=None
        z_range=[0.2,1.8]
        forward_passes=1
        skip_border_objects=False
        """
        Predict 6-DoF grasp distribution for given model and input data
        
        :param global_config: config.yaml from checkpoint directory
        :param checkpoint_dir: checkpoint directory
        :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
        :param K: Camera Matrix with intrinsics to convert depth to point cloud
        :param local_regions: Crop 3D local regions around given segments. 
        :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
        :param filter_grasps: Filter and assign grasp contacts according to segmap.
        :param segmap_id: only return grasps from specified segmap_id.
        :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
        :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
        """       
        with self.lock:
            rgb = self.rgb
            depth = self.depth
            info = self.info

        np.save("/tmp/contact_graspnet.npy", (rgb, depth, info, segmap))

        segmap = (segmap)
        segmap[segmap > 0] = 1
        masked_rgb = np.copy(rgb)
        masked_rgb[segmap==0] = 0
        #convert to meters
        depth = depth/1000 

        if segmap is None and (local_regions or filter_grasps):
            raise ValueError('Need segmentation map to extract local regions or filter grasps')

        info.K = np.asarray(info.K).reshape(3,3) 
        print(segmap.shape, depth.shape, rgb.shape)
        pc_full = None
        if pc_full is None:
            print('Converting depth to point cloud(s)...')
            pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(depth, info.K, segmap=segmap, rgb=rgb,
                                                                                    skip_border_objects=skip_border_objects, z_range=z_range)
            theta = radians(CAMERA_ROT)
            # sRotate the pointcloud, because the camera is sideways
            #rotation_matrix = np.asarray([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
            rotation_matrix = euler_matrix(0, 0, radians(CAMERA_ROT))[0:3, 0:3]

            print(rotation_matrix.shape, pc_full.shape)
            np.save("/tmp/contact_graspnet_pc_orig.npy", (pc_full, pc_colors))
            pc_full = pc_full@rotation_matrix
            for k in pc_segments:
                pc_segments[k] = pc_segments[k]@rotation_matrix

        print('Generating Grasps...')
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(self.sess, pc_full, pc_segments=pc_segments, 
                                                                                        local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  
        # Visualize results     
        
        top_grasp = pred_grasps_cam[1][np.argmax(scores[1])] 
        print(top_grasp)
        pose = PoseStamped()
        pose.header.frame_id = info.header.frame_id
        print(pose.header.frame_id)
        pose.header.stamp = rospy.Time.now()

        (trans, quat) = self.transform_grasp(top_grasp)
        pose.pose.position.x = trans[0]
        pose.pose.position.y = trans[1]
        pose.pose.position.z = trans[2]
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        self.pose_pub.publish(pose)

        posearr = PoseArray()
        posearr.header.stamp = rospy.Time.now()
        posearr.header.frame_id = self.info.header.frame_id
        print(self.info.header.frame_id)
        posearr.poses = []
        for g in pred_grasps_cam[1]:
            pose2 = Pose()
            (trans, quat) = self.transform_grasp(g)
            pose2.position.x = trans[0]
            pose2.position.y = trans[1]
            pose2.position.z = trans[2]
            pose2.orientation.x = quat[0]
            pose2.orientation.y = quat[1]
            pose2.orientation.z = quat[2]
            pose2.orientation.w = quat[3]
            posearr.poses.append(pose2)
        self.posearr_pub.publish(posearr)

        # show_image(np.rot90(self.rgb, 3), np.rot90(segmap, 3))  
        np.save("/tmp/contact_graspnet_pc.npy", (pc_full, pred_grasps_cam, scores, pc_colors))
        np.save("/tmp/contact_graspnet_grasps_rot.npy", (pred_grasps_cam, scores))
        # visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)

        return True, "", [pose]
            
if __name__ == "__main__":
    rospy.init_node("contact_graspnet")

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--ckpt_dir', default=default_dir+'checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    # parser.add_argument('--np_path', default=default_dir+'test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    # parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    # parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    # parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    # parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    # parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    # parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    # parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    # parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    # parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    # FLAGS = parser.parse_args()

    
    cg_server = ContactGraspnetServer()
    if len(sys.argv) > 1:
        if sys.argv[1] == "file":
            cg_server.inference_from_file()
        elif sys.argv[1] == "ros":
            cg_server.inference_from_ros_manual()
    rospy.spin()
