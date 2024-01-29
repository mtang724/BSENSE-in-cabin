from isens_vtrigU import isens_vtrigU
import numpy as np
import os
import time
import vtrigU as vtrig
RBWs = [50]
# vtrig.VTRIG_U_TXMODE__HIGH_RATE, vtrig.VTRIG_U_TXMODE__LOW_RATE
scan_profiles = [vtrig.VTRIG_U_TXMODE__MED_RATE]

gt = 30 # respiration ground truth
degree = 0
round = 1

calibrate = False
len_frame = 300 # 30s -40s
situation = "" # rear_1.3m, rear_w_reflector
case = "radar_data"
distance = 0.5 # in m

curr_dir = os.path.dirname(os.path.abspath(__file__))

def rec2arr(rec):
    recArr = []
    for key in rec.keys():
        recArr.append(rec[key])
    return np.array(recArr)

if calibrate:
    for RBW in RBWs:
        for scan_profile in scan_profiles:
            scenario = '{}k_profile_{}_distance_{}_round_{}'.format(RBW, scan_profile, distance, round)
            data_dir = os.path.join(curr_dir, '../data/{}/{}'.format(case, scenario))
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            radar = isens_vtrigU()
            radar.scan_setup(rbw=RBW, scan_profile = scan_profile)
            calFrame = radar.scan_calibration()    
            np.save(os.path.join(data_dir, 'calibration.npy'), calFrame)
else:
    for RBW in RBWs:
        for scan_profile in scan_profiles:
            scenario = '{}k_profile_{}_distance_{}_round_{}'.format(RBW, scan_profile, distance, round)
            data_dir = os.path.join(curr_dir, '../data/{}/{}'.format(case, scenario))
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            radar = isens_vtrigU()

            radar.scan_setup(rbw=RBW, scan_profile = scan_profile)

            data_list = []
            record_time = time.time()
            yieldData = radar.yield_data(nframes=len_frame)
            data = list(yieldData)
            record_time = time.time() - record_time
            data_list += data

            recData = np.array([rec2arr(rec) for rec in data_list])


            print("Data collection finished in {} seconds".format(record_time))
            print("Data saved to {}".format(data_dir))

            cfg = {
                'start_freq': radar.start_freq,
                'stop_freq': radar.stop_freq,

                'n_freq': radar.n_freq,
                'rbw': radar.rbw,
                'scan_profile': radar.scan_profile,
                'txRxPairs': radar.txRxPairs,
                'freq': radar.freq,
                'nfft': radar.nfft,
                'dist_vec': radar.dist_vec,
                'collect_time': time.time(),
                'sample_time' : record_time/len_frame
            }

            print("Radar initialized with configuration:")
            print("Start frequency: {} MHz".format(cfg['start_freq']))
            print("Stop frequency: {} MHz".format(cfg['stop_freq']))
            print("RBW: {} kHz".format(cfg['rbw']))
            print()

            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            np.save(os.path.join(data_dir, 'recording.npy'), recData)
            np.save(os.path.join(data_dir, 'config.npy'), cfg)