# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import copy

from common.h36m_dataset import h36m_skeleton
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates

treadmill_cameras_intrinsic_params = [
    {
        'id': 'id',
        'center': [958.4115524291992, 554.9155349731445],
        'focal_length': [902.431755065918, 902.0129013061523],
        'radial_distortion': [0.8090701103210449, -2.929135799407959, 1.6386618614196777, 0.6825506091117859,
                              -2.759711503982544, 1.570199966430664],
        # 'radial_distortion': [0,0,0],
        'tangential_distortion': [0.0010503812227398157, -0.00010656243102857843],
        'res_w': 1080,
        'res_h': 1080,
        'azimuth': 70,  # Only used for visualization
    }
]


#
# treadmill_cameras_extrinsic_params = {
#   "subject_01": [
#     {
#       "translation": [
#         1302.5813050668842,
#         -191.25052934801784,
#         842.9971888697308
#       ],
#       "orientation": [
#         0.5013503379131667,
#         -0.5075620182355353,
#         -0.4987430094991014,
#         0.4922235739933664
#       ]
#     }
#   ],
#   "subject_02": [
#     {
#       "translation": [
#         1300.3626632942794,
#         -195.35895279562806,
#         839.0920717330171
#       ],
#       "orientation": [
#         0.5025177981437896,
#         -0.5075591096399154,
#         -0.4976392872246426,
#         0.4921531799966382
#       ]
#     }
#   ],
#   "subject_03": [
#     {
#       "translation": [
#         1303.601491590281,
#         -197.45680763069453,
#         839.0794408937977
#       ],
#       "orientation": [
#         0.5024708189769712,
#         -0.5088257876394146,
#         -0.4962629294245929,
#         0.49228294586412685
#       ]
#     }
#   ],
#   "subject_04": [
#     {
#       "translation": [
#         1296.4122292335135,
#         -192.02921739570274,
#         832.945296580106
#       ],
#       "orientation": [
#         0.5029630835964768,
#         -0.506621992935505,
#         -0.49675474582772317,
#         0.49355548351812983
#       ]
#     }
#   ],
#   "subject_05": [
#     {
#       "translation": [
#         1292.9196237311808,
#         -198.002787723187,
#         836.9262622111235
#       ],
#       "orientation": [
#         0.506452217693634,
#         -0.5110972962760301,
#         -0.49278522153772836,
#         0.48933468134452107
#       ]
#     }
#   ],
#   "subject_06": [
#     {
#       "translation": [
#         2050.4817508168735,
#         -94.31775855579897,
#         960.7091167555923
#       ],
#       "orientation": [
#         0.486223784971226,
#         -0.5202437692318886,
#         -0.5051721102359134,
#         0.4875797273712747
#       ]
#     }
#   ],
#   "subject_07": [
#     {
#       "translation": [
#         2213.543006135151,
#         270.13942235052616,
#         1015.8635147967987
#       ],
#       "orientation": [
#         0.4961886566974292,
#         -0.5095056334895827,
#         -0.4951890809475692,
#         0.4989875755141421
#       ]
#     }
#   ],
#   "subject_08": [
#     {
#       "translation": [
#         2217.005408280264,
#         268.2476390286334,
#         1012.7088065447557
#       ],
#       "orientation": [
#         0.4972781897578881,
#         -0.5088461182260906,
#         -0.49490500355692796,
#         0.49885776270563764
#       ]
#     }
#   ],
#   "subject_09": [
#     {
#       "translation": [
#         2207.4336788892533,
#         270.3235715659189,
#         1011.4582397238968
#       ],
#       "orientation": [
#         0.49619987851919584,
#         -0.5091084485363611,
#         -0.5008672167892024,
#         0.4936864382705613
#       ]
#     }
#   ],
#   "subject_10": [
#     {
#       "translation": [
#         2219.50497330929,
#         278.4054181436622,
#         1011.2136064662594
#       ],
#       "orientation": [
#         0.4932347268569347,
#         -0.5113302454429038,
#         -0.4992162582369886,
#         0.4960282369276222
#       ]
#     }
#   ],
#   "subject_11": [
#     {
#       "translation": [
#         2192.0814290896396,
#         149.7147147176255,
#         1012.8496222400618
#       ],
#       "orientation": [
#         0.49989437891477556,
#         -0.5152403164376338,
#         -0.4935370580945489,
#         0.49097270650621294
#       ]
#     }
#   ],
#   "subject_12": [
#     {
#       "translation": [
#         2198.3875948461405,
#         214.16698751733142,
#         890.9602729640452
#       ],
#       "orientation": [
#         0.5017180750563494,
#         -0.515965486534257,
#         -0.49218837537802124,
#         0.4897031682662091
#       ]
#     }
#   ],
#   "subject_13": [
#     {
#       "translation": [
#         2198.6759820430652,
#         202.94866892522458,
#         898.1564361914842
#       ],
#       "orientation": [
#         0.4878598150740304,
#         -0.5062727095441031,
#         -0.5049832388768876,
#         0.5006722209791687
#       ]
#     }
#   ],
#   "subject_14": [
#     {
#       "translation": [
#         2212.3505660657665,
#         250.03026445585496,
#         895.6648096098966
#       ],
#       "orientation": [
#         0.49309048036679914,
#         -0.5094952396521161,
#         -0.4981510167151662,
#         0.4991211711489686
#       ]
#     }
#   ],
#   "subject_15": [
#     {
#       "translation": [
#         2221.8674032187864,
#         244.86806177344462,
#         893.5247840961798
#       ],
#       "orientation": [
#         0.4961543558160073,
#         -0.5143000181032125,
#         -0.4931095856193789,
#         0.49615449524731975
#       ]
#     }
#   ],
#   "subject_16": [
#     {
#       "translation": [
#         2215.727379823293,
#         274.35086025540664,
#         892.0430608130818
#       ],
#       "orientation": [
#         0.4936680013043616,
#         -0.5085978131106343,
#         -0.49791546069588766,
#         0.4997002731510551
#       ]
#     }
#   ]
# }

class TreadmillDataset(MocapDataset):
    def __init__(self, path):

        # Load serialized dataset
        data_3d = np.load(path, allow_pickle=True)
        metadata_3d = data_3d['metadata'].item()
        skeleton_data = metadata_3d['skeleton']
        skeleton = Skeleton(parents=skeleton_data["parents"], joints_left=skeleton_data["joints_left"],
                            joints_right=skeleton_data["joints_right"])
        super().__init__(fps=30, skeleton=skeleton)

        self._cameras = copy.deepcopy(metadata_3d["cameras_extrinsic_params"])
        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                cam.update(treadmill_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')

                # Normalize camera frame
                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype(
                    'float32')
                cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2
                if 'translation' in cam:
                    cam['translation'] = cam['translation'] / 1000  # mm to meters

                # Add intrinsic parameters vector
                cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                                   cam['center'],
                                                   cam['radial_distortion'],
                                                   cam['tangential_distortion']))

        # Load serialized dataset
        positions_3d = data_3d['positions_3d'].item()

        self._data = {}
        for subject, actions in positions_3d.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    'positions': positions,
                    'cameras': self._cameras[subject],
                }

    def supports_semi_supervised(self):
        return True
