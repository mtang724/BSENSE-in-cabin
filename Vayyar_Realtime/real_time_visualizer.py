import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QCheckBox, QComboBox
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import os
from parameter_setup import load_params, normalization
import time
from PIL import Image
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import find_peaks
from scipy.interpolate import RegularGridInterpolator
from cal_method import calibrate_pro_array

class RealDataVisualizer:
    def __init__(self) -> None:
        self.ntarget = 10
        self.enhance_rate = 1
        self.bound = 3

    def init_figure(self, root_dir, case_name, background_case):
        # load parameters
        self.data_path = "{}/{}/data_queue".format(root_dir, case_name)
        self.params_path = "{}/{}/parameters".format(root_dir, case_name)
        self.params = load_params(self.params_path)
        print('Start Processing & Visualizing...')
        vmin = 0.0#6
        cal_frame_name = sorted(os.listdir(self.data_path))[0]
        self.cal_frame = np.load(os.path.join(self.params_path, 'cal_arr.npy'))
        self.first_frame = np.load(os.path.join(self.data_path, cal_frame_name))
        # self.cal_frame = np.load(os.path.join(self.data_path, cal_frame_name))
        # 
        self.background_path = "{}/{}/data_queue".format(root_dir, background_case)
        background_frames_file = sorted(os.listdir(self.background_path))
        background_frames = [np.load(os.path.join(self.background_path, file)) for file in background_frames_file]
        self.background_frame = np.mean(background_frames, axis=0)
        self.data_queue = sorted(os.listdir(self.data_path))
        # self.data_queue = sorted(os.listdir(os.path.join(self.root_dir, self.case_name, 'data_queue')))
            # calArr = np.load(os.path.join(self.root_dir, self.case_name, 'parameters', "cal_arr.npy"))
        self.range_Nfft = int(np.load(os.path.join(root_dir, case_name, 'parameters', "range_Nfft.npy")))

        # First frame
        data_frame = self.data_queue[0]
        cur_frame_data = data_frame
        self.pro_arr = np.load(os.path.join(self.data_path, cur_frame_data))
        self.pro_arr = self.pro_arr#- cal_arr
        range_profile = np.linalg.norm(self.pro_arr,axis=0)

        # # load 
        range_profile = np.linalg.norm(np.fft.ifft(self.pro_arr, n=512, axis=1),axis=0)
        range_profile[np.where(self.params['dist_vec']>self.bound)]=np.min(np.abs(range_profile))
        raw_range_profile = np.linalg.norm(np.fft.ifft(self.pro_arr, n=512, axis=1),axis=0)
        raw_range_profile[np.where(self.params['dist_vec']>self.bound)]=np.min(np.abs(raw_range_profile))
        cal_range_profile = np.linalg.norm(np.fft.ifft(self.cal_frame, n=512, axis=1),axis=0)
        cal_range_profile[np.where(self.params['dist_vec']>self.bound)]=np.min(np.abs(cal_range_profile))

        self.pro_arr_3D = self.pro_arr.reshape(20,20,150)#[chosen_frame,:,:,:]
        self.pro_arr_3D = np.fft.ifft(self.pro_arr_3D, n=self.params['range_Nfft'], axis=2)
        self.pro_arr_3D[:,:,np.where(self.params['dist_vec']>self.bound)]=np.min(np.abs(self.pro_arr_3D))

        self.pro_arr_3D = np.fft.fft2(self.pro_arr_3D, s=self.params['angle_Nfft'], axes=[0,1])

        target_range_bin = np.argsort(range_profile)[::-1][:self.ntarget]#params['dist_vec'][np.argmax(range_profile)]

        # Define the coordinate values as per your actual data
        theta_values = self.params['AoD_vec']
        phi_values = self.params['AoA_vec']
        r_values = self.params['dist_vec']  # replace with actual range if different

        theta, phi, r = np.meshgrid(theta_values, phi_values, r_values, indexing='ij')

        x, y, z = self.spherical_to_rectangular(r, theta, phi)

        # Interpolate data to rectangular coordinates.
        interpolator = RegularGridInterpolator((theta_values, phi_values, r_values), self.pro_arr_3D)

        # Create a grid in spherical coordinates
        grid_theta, grid_phi, grid_r = np.meshgrid(theta_values, phi_values, r_values, indexing='ij')

        # Convert the grid to Cartesian coordinates
        grid_x, grid_y, grid_z = self.spherical_to_rectangular(grid_r, grid_theta, grid_phi)

        rect_data = interpolator((grid_theta, grid_phi, grid_r))

        # Project along the xy, xz, and yz planes
        xy_projection = np.linalg.norm(rect_data[:,:,target_range_bin], axis=2)#np.linalg.norm()#np.abs(rect_data[:,:,target_range_bin].mean(axis=2))
        xz_projection = np.linalg.norm(rect_data, axis=1)#np.abs(rect_data.mean(axis=1))
        yz_projection = np.linalg.norm(rect_data, axis=0)#np.abs(rect_data.mean(axis=0))
        # xy_projection = np.sum(np.abs(rect_data[:,:,target_range_bin]), axis=2)#np.abs(rect_data[:,:,target_range_bin].mean(axis=2))
        # xz_projection = np.abs(np.sum(rect_data, axis=1))#np.abs(rect_data.mean(axis=1))
        # yz_projection = np.abs(np.sum(rect_data, axis=0))#np.abs(rect_data.mean(axis=0))

        xy_projection = np.roll(xy_projection,shift=self.params['y_offset_shift'],axis=1)
        xy_projection = np.roll(xy_projection,shift=self.params['x_offset_shift'],axis=0)
        xz_projection = np.roll(xz_projection,shift=self.params['x_offset_shift'],axis=0)[:,np.where(self.params['dist_vec']<=self.bound)].squeeze()
        yz_projection = np.roll(yz_projection,shift=self.params['y_offset_shift'],axis=0)[:,np.where(self.params['dist_vec']<=self.bound)].squeeze()

        # peak_indices = filter_and_cluster(xy_projection, threshold=0.02, n=4)
        x_axis = np.linspace(grid_x.min(), grid_x.max(), 64)
        y_axis = np.linspace(grid_y.min(), grid_y.max(), 64)
        z_axis = np.linspace(grid_z.min(), grid_z.max(), 512)

        # self.extent = [np.min(self.params['AoD_vec']), np.max(self.params['AoD_vec']), np.min(self.params['AoA_vec']), np.max(self.params['AoA_vec'])]

        fig = plt.figure(figsize=(12,10))
        self.ax1 = plt.subplot(2,2,2)
        plt.title('XY Perspective')
        self.extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
        self.im1 = plt.imshow((xy_projection).T,origin='lower',aspect='auto', extent=self.extent, vmin=vmin, interpolation='nearest')
        self.colorbar1 = plt.colorbar()
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.grid()
        self.ax2 = plt.subplot(2,2,1)
        plt.title('{} Range Profile'.format(case_name))
        range_profile = np.linalg.norm(np.fft.ifft(self.pro_arr, n=512, axis=1),axis=0)
        range_profile[np.where(self.params['dist_vec']>3)]=np.mean((range_profile))
        self.plt1, = plt.plot(self.params['dist_vec'],range_profile)
        plt.xlabel('Range [m]')
        plt.ylabel('Magnitude')
        self.ax2_initial_xlim = self.ax2.get_xlim()
        self.ax2_initial_ylim = self.ax2.get_ylim()
        plt.grid()
        self.ax3 = plt.subplot(2,2,4)
        plt.title('YZ Perspective')
        self.extent  = [grid_y.min(), grid_y.max(), grid_z.min(), self.bound]
        self.im2 = plt.imshow((yz_projection).T,origin='lower',aspect='auto', extent=self.extent, vmin=vmin, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Y [m]')
        plt.ylabel('Z [m]')
        plt.grid()
        self.ax4 = plt.subplot(2,2,3)
        plt.title('XZ Perspective')
        self.extent  = [grid_x.min(), grid_x.max(), grid_z.min(), self.bound]
        self.im3 = plt.imshow((xz_projection).T,origin='lower',aspect='auto', extent=self.extent, vmin=vmin, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('X [m]')
        plt.ylabel('Z [m]')
        plt.grid()
        # plt.show()
            # plt.savefig('simulation_visualization.jpg')

        # anim = FuncAnimation(fig, self.update, frames=200, interval=200, blit=True)
        # f = r"./data/{}_animation.gif".format(case_name)
        # writergif = PillowWriter(fps=100) 
        # anim.save(f, writer=writergif)
        # Show the animation
        # plt.show()
        return fig, self.data_queue

    def update(self):
        # Modify the data for the imshow plot (example: add some random noise)
        # data_frame = self.data_queue[frame]
        # cur_frame_data = data_frame
        # rec_arr = np.load(os.path.join(self.data_path, cur_frame_data))
        range_profile = np.linalg.norm(self.pro_arr,axis=0)
        # # load 
        range_profile = np.linalg.norm(np.fft.ifft(self.pro_arr, n=512, axis=1),axis=0)
        range_profile[np.where(self.params['dist_vec']>self.bound)]=np.min(np.abs(range_profile))
        raw_range_profile = np.linalg.norm(np.fft.ifft(self.pro_arr, n=512, axis=1),axis=0)
        raw_range_profile[np.where(self.params['dist_vec']>self.bound)]=np.min(np.abs(raw_range_profile))
        cal_range_profile = np.linalg.norm(np.fft.ifft(self.cal_frame, n=512, axis=1),axis=0)
        cal_range_profile[np.where(self.params['dist_vec']>self.bound)]=np.min(np.abs(cal_range_profile))

        # self.pro_arr_3D = self.pro_arr.reshape(20,20,150)#[chosen_frame,:,:,:]
        # self.pro_arr_3D = np.fft.ifft(self.pro_arr_3D, n=self.params['range_Nfft'], axis=2)
        self.pro_arr_3D[:,:,np.where(self.params['dist_vec']>self.bound)]=np.min(np.abs(self.pro_arr_3D))

        self.pro_arr_3D = np.fft.fft2(self.pro_arr_3D, s=self.params['angle_Nfft'], axes=[0,1])

        target_range_bin = np.argsort(range_profile)[::-1][:self.ntarget]#params['dist_vec'][np.argmax(range_profile)]

        # Define the coordinate values as per your actual data
        theta_values = self.params['AoD_vec']
        phi_values = self.params['AoA_vec']
        r_values = self.params['dist_vec']  # replace with actual range if different

        theta, phi, r = np.meshgrid(theta_values, phi_values, r_values, indexing='ij')

        x, y, z = self.spherical_to_rectangular(r, theta, phi)

        # Interpolate data to rectangular coordinates.
        interpolator = RegularGridInterpolator((theta_values, phi_values, r_values), self.pro_arr_3D)

        # Create a grid in spherical coordinates
        grid_theta, grid_phi, grid_r = np.meshgrid(theta_values, phi_values, r_values, indexing='ij')

        # Convert the grid to Cartesian coordinates
        grid_x, grid_y, grid_z = self.spherical_to_rectangular(grid_r, grid_theta, grid_phi)

        rect_data = interpolator((grid_theta, grid_phi, grid_r))

        # Project along the xy, xz, and yz planes
        xy_projection = np.linalg.norm(rect_data[:,:,target_range_bin], axis=2)#np.linalg.norm()#np.abs(rect_data[:,:,target_range_bin].mean(axis=2))
        xz_projection = np.linalg.norm(rect_data, axis=1)#np.abs(rect_data.mean(axis=1))
        yz_projection = np.linalg.norm(rect_data, axis=0)#np.abs(rect_data.mean(axis=0))

        xy_projection = np.roll(xy_projection,shift=self.params['y_offset_shift'],axis=1)
        xy_projection = np.roll(xy_projection,shift=self.params['x_offset_shift'],axis=0)
        xz_projection = np.roll(xz_projection,shift=self.params['x_offset_shift'],axis=0)[:,np.where(self.params['dist_vec']<=self.bound)].squeeze()
        yz_projection = np.roll(yz_projection,shift=self.params['y_offset_shift'],axis=0)[:,np.where(self.params['dist_vec']<=self.bound)].squeeze()

        # self.extent = [np.min(self.params['AoD_vec']), np.max(self.params['AoD_vec']), np.min(self.params['AoA_vec']), np.max(self.params['AoA_vec'])]
        self.im1.set_data((xy_projection).T)
        self.im1.set_clim(vmin=xy_projection.min(), vmax=xy_projection.max())
        range_profile = np.linalg.norm(np.fft.ifft(self.pro_arr, n=512, axis=1),axis=0)
        range_profile[np.where(self.params['dist_vec']>3)]=np.mean((range_profile))
        self.plt1.set_ydata(range_profile)
        self.im2.set_data((yz_projection).T)
        self.im2.set_clim(vmin=yz_projection.min(), vmax=yz_projection.max())
        self.im3.set_data((xz_projection).T)
        self.im3.set_clim(vmin=xz_projection.min(), vmax=xz_projection.max())

        return [self.im1, self.plt1, self.im2, self.im3]  # Return the artists that need to be updated
    
    def spherical_to_rectangular(self, r, theta, phi):
        theta = np.radians(theta)
        phi = np.radians(phi)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

class AnimationWindow(QWidget):
    def __init__(self, folder_path):
        super().__init__()
        self.case_name = folder_path.split("/")[-1]
        self.root_dir = "/".join(folder_path.split("/")[:-1])
        background_case = self.case_name
        self.vis = RealDataVisualizer()
        self.figure, self.data_queue = self.vis.init_figure(self.root_dir, self.case_name, background_case)
        
        self.pro_arr = self.vis.pro_arr
        self.cal_frame = self.vis.cal_frame
        self.first_frame = self.vis.first_frame

        self.canvas = FigureCanvas(self.figure)

        # self.checkbox = QCheckBox("Change Data")
        self.comboBox = QComboBox()
        self.comboBox.addItems(["Raw", "Calibration Frame", "First Frame", "Cal-Average Range Profile", "First Frame-Average Range Profile"])
        self.checkBox = QCheckBox("Resize Axis")
        # self.checkbox.clicked.connect(self.update_data)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.comboBox)
        layout.addWidget(self.checkBox)
        self.setLayout(layout)

        self.ani = FuncAnimation(self.figure, self.update, frames=range(200), interval=200, blit=True)

        self.canvas.draw()

    def update(self, i):
        cur_frame_data = self.data_queue[i]
        self.pro_arr = np.load(os.path.join(self.root_dir, self.case_name, 'data_queue', cur_frame_data))
        index = self.comboBox.currentIndex()

        if index == 0:
            self.vis.pro_arr = self.pro_arr
            self.vis.pro_arr_3D = self.vis.pro_arr.reshape(20,20,150)#[chosen_frame,:,:,:]
            self.vis.pro_arr_3D = np.fft.ifft(self.vis.pro_arr_3D, n=self.vis.range_Nfft, axis=2)
        if index == 1:
            self.vis.pro_arr = self.pro_arr - self.cal_frame
            self.vis.pro_arr_3D = self.vis.pro_arr.reshape(20,20,150)#[chosen_frame,:,:,:]
            self.vis.pro_arr_3D = np.fft.ifft(self.vis.pro_arr_3D, n=self.vis.range_Nfft, axis=2)
        if index == 2:
            self.vis.pro_arr = self.pro_arr - self.first_frame
            self.vis.pro_arr_3D = self.vis.pro_arr.reshape(20,20,150)#[chosen_frame,:,:,:]
            self.vis.pro_arr_3D = np.fft.ifft(self.vis.pro_arr_3D, n=self.vis.range_Nfft, axis=2)
        if index == 3:
            self.avg_range_cal, _ = calibrate_pro_array(self.root_dir, self.case_name, self.data_queue, self.cal_frame, self.vis.range_Nfft, i, 20)
            # self.vis.pro_arr = self.pro_arr - self.avg_range_cal
            self.vis.pro_arr_3D = self.pro_arr.reshape(20,20,150)#[chosen_frame,:,:,:]
            self.vis.pro_arr_3D = np.fft.ifft(self.vis.pro_arr_3D, n=self.vis.range_Nfft, axis=2)
            self.vis.pro_arr_3D = self.vis.pro_arr_3D - self.avg_range_cal.reshape(20,20,-1)
        if index == 4:
            print("Not implemented yet")

        if self.checkBox.isChecked():
            self.vis.ax2.set_autoscaley_on(True)
            self.vis.ax2.relim()
            self.vis.ax2.autoscale_view()
        else:
            self.vis.ax2.set_ylim(self.vis.ax2_initial_ylim)
            self.vis.ax2.set_xlim(self.vis.ax2_initial_xlim)
            self.vis.ax2.set_autoscaley_on(False)

        return_plots = self.vis.update()
        return return_plots

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Main Window")

        self.ani_button = QPushButton("Open Animation Window")
        self.ani_button.clicked.connect(self.open_animation_window)

        layout = QVBoxLayout()
        layout.addWidget(self.ani_button)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        self.animation_window = None

    def open_animation_window(self):
        dialog = QFileDialog()
        
        # this will open the file dialog at the user's home directory
        home_dir = ""
        selected_folder = dialog.getExistingDirectory(self, "Select Folder", home_dir)

        if selected_folder:  # if a folder is selected
            if self.animation_window is None:
                self.animation_window = AnimationWindow(selected_folder)
            self.animation_window.show()

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()