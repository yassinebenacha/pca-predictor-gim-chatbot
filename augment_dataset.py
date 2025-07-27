"""
Script pour augmenter le dataset et supprimer la colonne "Statut GIM"
Génère plus de données synthétiques pour améliorer les performances du modèle
Auteur: Assistant IA
Date: 2025-07-26
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict
import itertools


class DatasetAugmenter:
    """
    Classe pour augmenter le dataset de diagnostic automobile
    """

    def __init__(self):
        """Initialise l'augmenteur avec des données de base"""

        # Codes DTC étendus
        self.dtc_codes = [
            'P0100', 'P0101', 'P0102', 'P0103', 'P0104', 'P0105', 'P0106', 'P0107', 'P0108', 'P0109',
            'P0110', 'P0111', 'P0112', 'P0113', 'P0114', 'P0115', 'P0116', 'P0117', 'P0118', 'P0119',
            'P0120', 'P0121', 'P0122', 'P0123', 'P0124', 'P0125', 'P0126', 'P0127', 'P0128', 'P0129',
            'P0130', 'P0131', 'P0132', 'P0133', 'P0134', 'P0135', 'P0136', 'P0137', 'P0138', 'P0139',
            'P0140', 'P0141', 'P0142', 'P0143', 'P0144', 'P0145', 'P0146', 'P0147', 'P0148', 'P0149',
            'P0150', 'P0151', 'P0152', 'P0153', 'P0154', 'P0155', 'P0156', 'P0157', 'P0158', 'P0159',
            'P0160', 'P0161', 'P0162', 'P0163', 'P0164', 'P0165', 'P0166', 'P0167', 'P0168', 'P0169',
            'P0170', 'P0171', 'P0172', 'P0173', 'P0174', 'P0175', 'P0176', 'P0177', 'P0178', 'P0179',
            'P0180', 'P0181', 'P0182', 'P0183', 'P0184', 'P0185', 'P0186', 'P0187', 'P0188', 'P0189',
            'P0190', 'P0191', 'P0192', 'P0193', 'P0194', 'P0195', 'P0196', 'P0197', 'P0198', 'P0199',
            'P0200', 'P0201', 'P0202', 'P0203', 'P0204', 'P0205', 'P0206', 'P0207', 'P0208', 'P0209',
            'P0210', 'P0211', 'P0212', 'P0213', 'P0214', 'P0215', 'P0216', 'P0217', 'P0218', 'P0219',
            'P0220', 'P0221', 'P0222', 'P0223', 'P0224', 'P0225', 'P0226', 'P0227', 'P0228', 'P0229',
            'P0230', 'P0231', 'P0232', 'P0233', 'P0234', 'P0235', 'P0236', 'P0237', 'P0238', 'P0239',
            'P0240', 'P0241', 'P0242', 'P0243', 'P0244', 'P0245', 'P0246', 'P0247', 'P0248', 'P0249',
            'P0250', 'P0251', 'P0252', 'P0253', 'P0254', 'P0255', 'P0256', 'P0257', 'P0258', 'P0259',
            'P0260', 'P0261', 'P0262', 'P0263', 'P0264', 'P0265', 'P0266', 'P0267', 'P0268', 'P0269',
            'P0270', 'P0271', 'P0272', 'P0273', 'P0274', 'P0275', 'P0276', 'P0277', 'P0278', 'P0279',
            'P0280', 'P0281', 'P0282', 'P0283', 'P0284', 'P0285', 'P0286', 'P0287', 'P0288', 'P0289',
            'P0290', 'P0291', 'P0292', 'P0293', 'P0294', 'P0295', 'P0296', 'P0297', 'P0298', 'P0299',
            'P0300', 'P0301', 'P0302', 'P0303', 'P0304', 'P0305', 'P0306', 'P0307', 'P0308', 'P0309',
            'P0310', 'P0311', 'P0312', 'P0313', 'P0314', 'P0315', 'P0316', 'P0317', 'P0318', 'P0319',
            'P0320', 'P0321', 'P0322', 'P0323', 'P0324', 'P0325', 'P0326', 'P0327', 'P0328', 'P0329',
            'P0330', 'P0331', 'P0332', 'P0333', 'P0334', 'P0335', 'P0336', 'P0337', 'P0338', 'P0339',
            'P0340', 'P0341', 'P0342', 'P0343', 'P0344', 'P0345', 'P0346', 'P0347', 'P0348', 'P0349',
            'P0350', 'P0351', 'P0352', 'P0353', 'P0354', 'P0355', 'P0356', 'P0357', 'P0358', 'P0359',
            'P0360', 'P0361', 'P0362', 'P0363', 'P0364', 'P0365', 'P0366', 'P0367', 'P0368', 'P0369',
            'P0370', 'P0371', 'P0372', 'P0373', 'P0374', 'P0375', 'P0376', 'P0377', 'P0378', 'P0379',
            'P0380', 'P0381', 'P0382', 'P0383', 'P0384', 'P0385', 'P0386', 'P0387', 'P0388', 'P0389',
            'P0390', 'P0391', 'P0392', 'P0393', 'P0394', 'P0395', 'P0396', 'P0397', 'P0398', 'P0399',
            'P0400', 'P0401', 'P0402', 'P0403', 'P0404', 'P0405', 'P0406', 'P0407', 'P0408', 'P0409',
            'P0410', 'P0411', 'P0412', 'P0413', 'P0414', 'P0415', 'P0416', 'P0417', 'P0418', 'P0419',
            'P0420', 'P0421', 'P0422', 'P0423', 'P0424', 'P0425', 'P0426', 'P0427', 'P0428', 'P0429',
            'P0430', 'P0431', 'P0432', 'P0433', 'P0434', 'P0435', 'P0436', 'P0437', 'P0438', 'P0439',
            'P0440', 'P0441', 'P0442', 'P0443', 'P0444', 'P0445', 'P0446', 'P0447', 'P0448', 'P0449',
            'P0450', 'P0451', 'P0452', 'P0453', 'P0454', 'P0455', 'P0456', 'P0457', 'P0458', 'P0459',
            'P0460', 'P0461', 'P0462', 'P0463', 'P0464', 'P0465', 'P0466', 'P0467', 'P0468', 'P0469',
            'P0470', 'P0471', 'P0472', 'P0473', 'P0474', 'P0475', 'P0476', 'P0477', 'P0478', 'P0479',
            'P0480', 'P0481', 'P0482', 'P0483', 'P0484', 'P0485', 'P0486', 'P0487', 'P0488', 'P0489',
            'P0490', 'P0491', 'P0492', 'P0493', 'P0494', 'P0495', 'P0496', 'P0497', 'P0498', 'P0499',
            'P0500', 'P0501', 'P0502', 'P0503', 'P0504', 'P0505', 'P0506', 'P0507', 'P0508', 'P0509',
            'P0510', 'P0511', 'P0512', 'P0513', 'P0514', 'P0515', 'P0516', 'P0517', 'P0518', 'P0519',
            'P0520', 'P0521', 'P0522', 'P0523', 'P0524', 'P0525', 'P0526', 'P0527', 'P0528', 'P0529',
            'P0530', 'P0531', 'P0532', 'P0533', 'P0534', 'P0535', 'P0536', 'P0537', 'P0538', 'P0539',
            'P0540', 'P0541', 'P0542', 'P0543', 'P0544', 'P0545', 'P0546', 'P0547', 'P0548', 'P0549',
            'P0550', 'P0551', 'P0552', 'P0553', 'P0554', 'P0555', 'P0556', 'P0557', 'P0558', 'P0559',
            'P0560', 'P0561', 'P0562', 'P0563', 'P0564', 'P0565', 'P0566', 'P0567', 'P0568', 'P0569',
            'P0570', 'P0571', 'P0572', 'P0573', 'P0574', 'P0575', 'P0576', 'P0577', 'P0578', 'P0579',
            'P0580', 'P0581', 'P0582', 'P0583', 'P0584', 'P0585', 'P0586', 'P0587', 'P0588', 'P0589',
            'P0590', 'P0591', 'P0592', 'P0593', 'P0594', 'P0595', 'P0596', 'P0597', 'P0598', 'P0599',
            'P0600', 'P0601', 'P0602', 'P0603', 'P0604', 'P0605', 'P0606', 'P0607', 'P0608', 'P0609',
            'P0610', 'P0611', 'P0612', 'P0613', 'P0614', 'P0615', 'P0616', 'P0617', 'P0618', 'P0619',
            'P0620', 'P0621', 'P0622', 'P0623', 'P0624', 'P0625', 'P0626', 'P0627', 'P0628', 'P0629',
            'P0630', 'P0631', 'P0632', 'P0633', 'P0634', 'P0635', 'P0636', 'P0637', 'P0638', 'P0639',
            'P0640', 'P0641', 'P0642', 'P0643', 'P0644', 'P0645', 'P0646', 'P0647', 'P0648', 'P0649',
            'P0650', 'P0651', 'P0652', 'P0653', 'P0654', 'P0655', 'P0656', 'P0657', 'P0658', 'P0659',
            'P0660', 'P0661', 'P0662', 'P0663', 'P0664', 'P0665', 'P0666', 'P0667', 'P0668', 'P0669',
            'P0670', 'P0671', 'P0672', 'P0673', 'P0674', 'P0675', 'P0676', 'P0677', 'P0678', 'P0679',
            'P0680', 'P0681', 'P0682', 'P0683', 'P0684', 'P0685', 'P0686', 'P0687', 'P0688', 'P0689',
            'P0690', 'P0691', 'P0692', 'P0693', 'P0694', 'P0695', 'P0696', 'P0697', 'P0698', 'P0699',
            'P0700', 'P0701', 'P0702', 'P0703', 'P0704', 'P0705', 'P0706', 'P0707', 'P0708', 'P0709',
            'P0710', 'P0711', 'P0712', 'P0713', 'P0714', 'P0715', 'P0716', 'P0717', 'P0718', 'P0719',
            'P0720', 'P0721', 'P0722', 'P0723', 'P0724', 'P0725', 'P0726', 'P0727', 'P0728', 'P0729',
            'P0730', 'P0731', 'P0732', 'P0733', 'P0734', 'P0735', 'P0736', 'P0737', 'P0738', 'P0739',
            'P0740', 'P0741', 'P0742', 'P0743', 'P0744', 'P0745', 'P0746', 'P0747', 'P0748', 'P0749',
            'P0750', 'P0751', 'P0752', 'P0753', 'P0754', 'P0755', 'P0756', 'P0757', 'P0758', 'P0759',
            'P0760', 'P0761', 'P0762', 'P0763', 'P0764', 'P0765', 'P0766', 'P0767', 'P0768', 'P0769',
            'P0770', 'P0771', 'P0772', 'P0773', 'P0774', 'P0775', 'P0776', 'P0777', 'P0778', 'P0779',
            'P0780', 'P0781', 'P0782', 'P0783', 'P0784', 'P0785', 'P0786', 'P0787', 'P0788', 'P0789',
            'P0790', 'P0791', 'P0792', 'P0793', 'P0794', 'P0795', 'P0796', 'P0797', 'P0798', 'P0799',
            'P0800', 'P0801', 'P0802', 'P0803', 'P0804', 'P0805', 'P0806', 'P0807', 'P0808', 'P0809',
            'P0810', 'P0811', 'P0812', 'P0813', 'P0814', 'P0815', 'P0816', 'P0817', 'P0818', 'P0819',
            'P0820', 'P0821', 'P0822', 'P0823', 'P0824', 'P0825', 'P0826', 'P0827', 'P0828', 'P0829',
            'P0830', 'P0831', 'P0832', 'P0833', 'P0834', 'P0835', 'P0836', 'P0837', 'P0838', 'P0839',
            'P0840', 'P0841', 'P0842', 'P0843', 'P0844', 'P0845', 'P0846', 'P0847', 'P0848', 'P0849',
            'P0850', 'P0851', 'P0852', 'P0853', 'P0854', 'P0855', 'P0856', 'P0857', 'P0858', 'P0859',
            'P0860', 'P0861', 'P0862', 'P0863', 'P0864', 'P0865', 'P0866', 'P0867', 'P0868', 'P0869',
            'P0870', 'P0871', 'P0872', 'P0873', 'P0874', 'P0875', 'P0876', 'P0877', 'P0878', 'P0879',
            'P0880', 'P0881', 'P0882', 'P0883', 'P0884', 'P0885', 'P0886', 'P0887', 'P0888', 'P0889',
            'P0890', 'P0891', 'P0892', 'P0893', 'P0894', 'P0895', 'P0896', 'P0897', 'P0898', 'P0899',
            'P0900', 'P0901', 'P0902', 'P0903', 'P0904', 'P0905', 'P0906', 'P0907', 'P0908', 'P0909',
            'P0910', 'P0911', 'P0912', 'P0913', 'P0914', 'P0915', 'P0916', 'P0917', 'P0918', 'P0919',
            'P0920', 'P0921', 'P0922', 'P0923', 'P0924', 'P0925', 'P0926', 'P0927', 'P0928', 'P0929',
            'P0930', 'P0931', 'P0932', 'P0933', 'P0934', 'P0935', 'P0936', 'P0937', 'P0938', 'P0939',
            'P0940', 'P0941', 'P0942', 'P0943', 'P0944', 'P0945', 'P0946', 'P0947', 'P0948', 'P0949',
            'P0950', 'P0951', 'P0952', 'P0953', 'P0954', 'P0955', 'P0956', 'P0957', 'P0958', 'P0959',
            'P0960', 'P0961', 'P0962', 'P0963', 'P0964', 'P0965', 'P0966', 'P0967', 'P0968', 'P0969',
            'P0970', 'P0971', 'P0972', 'P0973', 'P0974', 'P0975', 'P0976', 'P0977', 'P0978', 'P0979',
            'P0980', 'P0981', 'P0982', 'P0983', 'P0984', 'P0985', 'P0986', 'P0987', 'P0988', 'P0989',
            'P0990', 'P0991', 'P0992', 'P0993', 'P0994', 'P0995', 'P0996', 'P0997', 'P0998', 'P0999',
            'P1000', 'P1001', 'P1002', 'P1003', 'P1004', 'P1005', 'P1006', 'P1007', 'P1008', 'P1009',
            'P1010', 'P1011', 'P1012', 'P1013', 'P1014', 'P1015', 'P1016', 'P1017', 'P1018', 'P1019',
            'P1020', 'P1021', 'P1022', 'P1023', 'P1024', 'P1025', 'P1026', 'P1027', 'P1028', 'P1029',
            'P1030', 'P1031', 'P1032', 'P1033', 'P1034', 'P1035', 'P1036', 'P1037', 'P1038', 'P1039',
            'P1040', 'P1041', 'P1042', 'P1043', 'P1044', 'P1045', 'P1046', 'P1047', 'P1048', 'P1049',
            'P1050', 'P1051', 'P1052', 'P1053', 'P1054', 'P1055', 'P1056', 'P1057', 'P1058', 'P1059',
            'P1060', 'P1061', 'P1062', 'P1063', 'P1064', 'P1065', 'P1066', 'P1067', 'P1068', 'P1069',
            'P1070', 'P1071', 'P1072', 'P1073', 'P1074', 'P1075', 'P1076', 'P1077', 'P1078', 'P1079',
            'P1080', 'P1081', 'P1082', 'P1083', 'P1084', 'P1085', 'P1086', 'P1087', 'P1088', 'P1089',
            'P1090', 'P1091', 'P1092', 'P1093', 'P1094', 'P1095', 'P1096', 'P1097', 'P1098', 'P1099',
            # Codes B (Body), U (Network), C (Chassis)
            'B0001', 'B0002', 'B0003', 'B0004', 'B0005', 'B0010', 'B0015', 'B0020', 'B0025', 'B0030',
            'B1000', 'B1001', 'B1002', 'B1003', 'B1004', 'B1005', 'B1010', 'B1015', 'B1020', 'B1025',
            'U0001', 'U0002', 'U0003', 'U0004', 'U0005', 'U0010', 'U0015', 'U0020', 'U0025', 'U0030',
            'U1000', 'U1001', 'U1002', 'U1003', 'U1004', 'U1005', 'U1010', 'U1015', 'U1020', 'U1025',
            'C0001', 'C0002', 'C0003', 'C0004', 'C0005', 'C0010', 'C0015', 'C0020', 'C0025', 'C0030',
            'C1000', 'C1001', 'C1002', 'C1003', 'C1004', 'C1005', 'C1010', 'C1015', 'C1020', 'C1025'
        ]

        # Descriptions de problèmes étendues
        self.problem_descriptions = [
            # Moteur
            "Engine misfiring randomly", "Engine knocking sound", "Engine stalling at idle",
            "Engine overheating", "Engine rough idle", "Engine won't start", "Engine lacks power",
            "Engine making unusual noise", "Engine vibration excessive", "Engine oil leak",
            "Engine coolant leak", "Engine smoking", "Engine backfiring", "Engine hesitation",
            "Engine surging", "Engine running rich", "Engine running lean", "Engine timing off",
            "Engine compression low", "Engine valve noise", "Engine bearing noise", "Engine misfire cylinder 1",
            "Engine misfire cylinder 2", "Engine misfire cylinder 3", "Engine misfire cylinder 4",
            "Engine misfire multiple cylinders", "Engine fuel system malfunction", "Engine air intake problem",
            "Engine exhaust system issue", "Engine ignition system failure", "Engine fuel pump failure",
            "Engine fuel injector clogged", "Engine spark plug worn", "Engine coil failure",
            "Engine sensor malfunction", "Engine ECU error", "Engine timing chain stretched",
            "Engine camshaft position error", "Engine crankshaft position error", "Engine throttle body dirty",
            "Engine mass airflow sensor dirty", "Engine oxygen sensor failure", "Engine catalytic converter failure",

            # Transmission
            "Transmission slipping", "Transmission hard shifting", "Transmission won't shift",
            "Transmission fluid leak", "Transmission overheating", "Transmission noise",
            "Transmission vibration", "Transmission delayed engagement", "Transmission rough shifting",
            "Transmission stuck in gear", "Transmission no reverse", "Transmission no forward",
            "Transmission torque converter failure", "Transmission valve body issue", "Transmission solenoid failure",
            "Transmission clutch worn", "Transmission bands worn", "Transmission pump failure",
            "Transmission filter clogged", "Transmission cooler lines leak", "Transmission mount broken",
            "Transmission speed sensor failure", "Transmission range sensor error", "Transmission control module failure",

            # Freins
            "Brake pedal soft", "Brake pedal hard", "Brake noise squealing", "Brake noise grinding",
            "Brake pulling to one side", "Brake vibration", "Brake warning light on",
            "Brake fluid leak", "Brake pads worn", "Brake rotors warped", "Brake calipers sticking",
            "Brake master cylinder failure", "Brake booster failure", "Brake lines corroded",
            "Brake ABS malfunction", "Brake traction control issue", "Brake stability control problem",
            "Brake electronic system failure", "Brake parking brake stuck", "Brake emergency brake failure",

            # Suspension
            "Suspension noise", "Suspension rough ride", "Suspension bouncing", "Suspension sagging",
            "Suspension alignment off", "Suspension steering wheel vibration", "Suspension pulling",
            "Suspension uneven tire wear", "Suspension shock absorber leak", "Suspension strut failure",
            "Suspension spring broken", "Suspension ball joint worn", "Suspension tie rod worn",
            "Suspension control arm bushing worn", "Suspension sway bar link broken", "Suspension mount failure",

            # Électrique
            "Battery dead", "Battery won't hold charge", "Alternator not charging", "Starter won't engage",
            "Electrical system malfunction", "Lights not working", "Horn not working", "Wipers not working",
            "Power windows not working", "Power locks not working", "Radio not working", "AC not working",
            "Heater not working", "Electrical short circuit", "Fuse blown", "Relay failure",
            "Wiring harness damaged", "Electrical connector corroded", "Ground connection poor",
            "Electrical noise interference", "Electrical component overheating", "Electrical system overload",

            # Climatisation
            "AC not cooling", "AC blowing warm air", "AC compressor noise", "AC refrigerant leak",
            "AC condenser blocked", "AC evaporator frozen", "AC expansion valve stuck", "AC clutch failure",
            "AC pressure switch failure", "AC temperature sensor error", "AC blend door stuck",
            "AC fan not working", "AC vent blocked", "AC system contaminated", "AC desiccant saturated",

            # Carburant
            "Fuel gauge inaccurate", "Fuel pump failure", "Fuel filter clogged", "Fuel injector dirty",
            "Fuel pressure low", "Fuel pressure high", "Fuel tank leak", "Fuel cap loose",
            "Fuel vapor leak", "Fuel system contaminated", "Fuel quality poor", "Fuel additive needed",

            # Échappement
            "Exhaust noise loud", "Exhaust smoke black", "Exhaust smoke white", "Exhaust smoke blue",
            "Exhaust system leak", "Exhaust pipe damaged", "Exhaust muffler failure", "Exhaust catalytic converter clogged",
            "Exhaust oxygen sensor failure", "Exhaust emissions high", "Exhaust backpressure high",

            # Refroidissement
            "Cooling system overheating", "Coolant leak", "Radiator clogged", "Thermostat stuck",
            "Water pump failure", "Cooling fan not working", "Coolant temperature sensor error",
            "Coolant level low", "Coolant contaminated", "Cooling system air trapped",

            # Direction
            "Steering hard", "Steering loose", "Steering noise", "Steering vibration",
            "Steering wheel off center", "Power steering fluid leak", "Power steering pump failure",
            "Steering rack failure", "Steering column noise", "Steering wheel play excessive",

            # Éclairage
            "Headlights dim", "Headlights not working", "Tail lights not working", "Turn signals not working",
            "Brake lights not working", "Hazard lights not working", "Interior lights not working",
            "Dashboard lights not working", "Light bulb burned out", "Light switch failure",

            # Carrosserie
            "Door won't open", "Door won't close", "Window won't go up", "Window won't go down",
            "Sunroof stuck", "Hood won't open", "Trunk won't open", "Mirror adjustment not working",
            "Seat adjustment not working", "Seat heater not working", "Door lock not working",

            # Hybride/Électrique
            "Hybrid battery failure", "Electric motor noise", "Regenerative braking issue",
            "Charging system malfunction", "High voltage system error", "Inverter failure",
            "DC-DC converter failure", "Hybrid system overheating", "Electric drive malfunction",

            # Capteurs
            "Sensor malfunction", "Sensor reading erratic", "Sensor no signal", "Sensor voltage high",
            "Sensor voltage low", "Sensor circuit open", "Sensor circuit short", "Sensor contaminated",
            "Sensor calibration needed", "Sensor replacement required",

            # Divers
            "Check engine light on", "ABS light on", "Airbag light on", "Oil pressure light on",
            "Temperature warning light on", "Battery warning light on", "Brake warning light on",
            "Tire pressure warning light on", "Service engine soon", "Maintenance required",
            "System malfunction", "Component failure", "Performance degraded", "Efficiency reduced",
            "Noise abnormal", "Vibration excessive", "Temperature abnormal", "Pressure abnormal",
            "Voltage abnormal", "Current abnormal", "Resistance abnormal", "Frequency abnormal"
        ]

        # Causes racines étendues
        self.root_causes = [
            # Usure normale
            "Component worn out", "Normal wear and tear", "End of service life", "Material fatigue",
            "Friction wear", "Corrosion damage", "Oxidation", "Thermal cycling damage",

            # Défauts de fabrication
            "Manufacturing defect", "Quality control issue", "Material defect", "Assembly error",
            "Design flaw", "Tolerance issue", "Specification error", "Production variance",

            # Maintenance
            "Lack of maintenance", "Overdue service", "Wrong fluid used", "Contaminated fluid",
            "Filter not changed", "Improper installation", "Incorrect adjustment", "Missing lubrication",

            # Environnement
            "Environmental damage", "Salt corrosion", "UV damage", "Temperature extremes",
            "Moisture damage", "Dust contamination", "Chemical exposure", "Road debris damage",

            # Utilisation
            "Abuse or misuse", "Overloading", "Excessive speed", "Hard driving conditions",
            "Stop and go traffic", "Short trips", "Towing heavy loads", "Racing conditions",

            # Électrique
            "Electrical short circuit", "Open circuit", "Poor connection", "Corrosion on terminals",
            "Voltage spike", "Ground fault", "Wiring damage", "Connector failure",

            # Mécanique
            "Mechanical failure", "Bearing failure", "Seal failure", "Gasket failure",
            "Spring failure", "Bolt loosening", "Alignment issue", "Clearance problem",

            # Fluides
            "Fluid leak", "Fluid contamination", "Wrong viscosity", "Fluid breakdown",
            "Air in system", "Pressure loss", "Flow restriction", "Pump failure",

            # Capteurs
            "Sensor drift", "Sensor contamination", "Sensor damage", "Calibration error",
            "Signal interference", "Wiring issue", "Connector corrosion", "Software bug",

            # Carburant
            "Poor fuel quality", "Water in fuel", "Fuel contamination", "Wrong octane rating",
            "Fuel system deposit", "Injector clogging", "Fuel pump wear", "Filter restriction",

            # Spécifiques
            "Faulty spark plugs", "Worn brake pads", "Clogged air filter", "Dirty throttle body",
            "Vacuum leak", "Exhaust leak", "Coolant leak", "Oil leak", "Transmission fluid leak",
            "Power steering fluid leak", "Brake fluid leak", "Fuel leak", "Refrigerant leak",
            "Turbocharger failure", "Supercharger failure", "Timing belt failure", "Timing chain stretch",
            "Valve adjustment needed", "Carbon buildup", "Fuel injector failure", "Ignition coil failure",
            "Oxygen sensor failure", "Mass airflow sensor failure", "Throttle position sensor failure",
            "Crankshaft position sensor failure", "Camshaft position sensor failure", "Knock sensor failure",
            "Temperature sensor failure", "Pressure sensor failure", "Speed sensor failure",
            "ABS sensor failure", "Wheel bearing failure", "CV joint failure", "Ball joint failure",
            "Tie rod failure", "Control arm bushing failure", "Shock absorber failure", "Strut failure",
            "Spring failure", "Sway bar link failure", "Motor mount failure", "Transmission mount failure",
            "Engine mount failure", "Exhaust mount failure", "Heat shield failure", "Catalytic converter failure",
            "Muffler failure", "Resonator failure", "EGR valve failure", "PCV valve failure",
            "Thermostat failure", "Water pump failure", "Radiator failure", "Cooling fan failure",
            "AC compressor failure", "AC condenser failure", "AC evaporator failure", "AC expansion valve failure",
            "Alternator failure", "Starter failure", "Battery failure", "Voltage regulator failure",
            "Fuse failure", "Relay failure", "Switch failure", "Motor failure", "Actuator failure",
            "Solenoid failure", "Valve failure", "Pump failure", "Compressor failure", "Fan failure",
            "Heater failure", "Cooler failure", "Filter failure", "Gasket failure", "Seal failure",
            "Bearing failure", "Bushing failure", "Joint failure", "Link failure", "Rod failure",
            "Arm failure", "Bracket failure", "Mount failure", "Support failure", "Housing failure",
            "Cover failure", "Cap failure", "Plug failure", "Connector failure", "Terminal failure",
            "Wire failure", "Cable failure", "Hose failure", "Tube failure", "Pipe failure",
            "Line failure", "Fitting failure", "Clamp failure", "Clip failure", "Fastener failure"
        ]

        # Composants étendus
        self.components = [
            "Engine", "Transmission", "Brake System", "Suspension", "Electrical System",
            "Air Conditioning", "Fuel System", "Exhaust System", "Cooling System", "Steering",
            "Lighting System", "Body Electronics", "Hybrid System", "Battery", "Alternator",
            "Starter", "Ignition System", "Fuel Injection", "Turbocharger", "Supercharger",
            "Timing System", "Valve Train", "Lubrication System", "Emission Control",
            "Drivetrain", "Differential", "Axle", "CV Joints", "Wheel Bearings",
            "Tires", "Wheels", "Brakes", "ABS", "Traction Control", "Stability Control",
            "Power Steering", "Steering Column", "Steering Rack", "Tie Rods", "Ball Joints",
            "Control Arms", "Shock Absorbers", "Struts", "Springs", "Sway Bars",
            "Bushings", "Mounts", "Exhaust Manifold", "Catalytic Converter", "Muffler",
            "Resonator", "Exhaust Pipes", "Heat Shields", "Radiator", "Water Pump",
            "Thermostat", "Cooling Fans", "Hoses", "Belts", "Pulleys", "Tensioners",
            "AC Compressor", "AC Condenser", "AC Evaporator", "AC Expansion Valve",
            "AC Lines", "AC Refrigerant", "Heater Core", "Blower Motor", "Climate Control",
            "Fuel Tank", "Fuel Pump", "Fuel Filter", "Fuel Lines", "Fuel Rail",
            "Fuel Injectors", "Fuel Pressure Regulator", "Carbon Canister", "Purge Valve",
            "Headlights", "Tail Lights", "Turn Signals", "Brake Lights", "Hazard Lights",
            "Interior Lights", "Dashboard", "Gauges", "Warning Lights", "Switches",
            "Relays", "Fuses", "Wiring Harness", "Connectors", "Sensors", "Actuators",
            "ECU", "PCM", "BCM", "TCM", "ABS Module", "Airbag Module", "Immobilizer",
            "Keyless Entry", "Remote Start", "Navigation System", "Audio System",
            "Bluetooth", "USB Ports", "Power Outlets", "Seats", "Seat Belts", "Airbags",
            "Doors", "Windows", "Mirrors", "Sunroof", "Hood", "Trunk", "Tailgate",
            "Bumpers", "Fenders", "Quarter Panels", "Roof", "Pillars", "Frame",
            "Unibody", "Chassis", "Subframe", "Crossmember", "Reinforcements",
            "Hybrid Battery", "Electric Motor", "Inverter", "DC-DC Converter",
            "Charging System", "Regenerative Braking", "High Voltage System",
            "Low Voltage System", "12V Battery", "Voltage Regulator", "Ground Straps"
        ]

        # Solutions PCA étendues et réalistes
        self.pca_solutions = [
            # Moteur
            "Replace spark plugs and clean fuel injectors",
            "Replace ignition coil in cylinder 1",
            "Replace ignition coil in cylinder 2",
            "Replace ignition coil in cylinder 3",
            "Replace ignition coil in cylinder 4",
            "Replace oxygen sensor and clear fault codes",
            "Clean throttle body and reset ECU",
            "Replace mass airflow sensor",
            "Replace crankshaft position sensor",
            "Replace camshaft position sensor",
            "Replace thermostat and flush cooling system",
            "Replace water pump and refill coolant",
            "Replace timing belt and tensioner",
            "Replace timing chain and guides",
            "Replace fuel pump and filter",
            "Clean fuel injectors and replace filter",
            "Replace catalytic converter",
            "Replace turbocharger and oil lines",
            "Replace engine mount",
            "Perform engine compression test and repair",
            "Replace valve cover gasket",
            "Replace head gasket",
            "Replace connecting rod bearings",
            "Rebuild engine top end",
            "Replace engine oil pump",

            # Transmission
            "Replace clutch friction element and recalibrate transmission",
            "Flush transmission fluid and replace filter",
            "Replace transmission solenoid pack",
            "Replace transmission valve body",
            "Replace torque converter",
            "Replace transmission mount",
            "Repair transmission leak and refill",
            "Replace transmission speed sensor",
            "Replace transmission range sensor",
            "Update transmission control module software",
            "Replace transmission cooler lines",
            "Rebuild transmission",

            # Freins
            "Replace brake pads and resurface rotors",
            "Replace brake rotors and pads",
            "Replace brake calipers and bleed system",
            "Replace brake master cylinder",
            "Replace brake booster",
            "Flush brake fluid and bleed system",
            "Replace brake lines and fittings",
            "Replace ABS module and calibrate",
            "Replace brake pedal position sensor",
            "Replace parking brake cables",

            # Suspension
            "Replace shock absorbers",
            "Replace struts and strut mounts",
            "Replace coil springs",
            "Replace ball joints",
            "Replace tie rod ends",
            "Replace control arm bushings",
            "Replace sway bar links",
            "Perform wheel alignment",
            "Replace wheel bearings",
            "Replace suspension mount",

            # Électrique
            "Replace battery and clean terminals",
            "Clean battery terminals and apply protectant",
            "Replace alternator and belt",
            "Replace starter motor",
            "Replace voltage regulator",
            "Repair wiring harness",
            "Replace fuse box",
            "Replace relay",
            "Replace switch",
            "Repair electrical connector",

            # Climatisation
            "Locate and repair refrigerant leak, then recharge AC system",
            "Replace AC compressor and receiver dryer",
            "Replace AC condenser and flush system",
            "Replace AC evaporator and cabin filter",
            "Replace AC expansion valve",
            "Replace AC pressure switch",
            "Replace blower motor and resistor",
            "Replace cabin air filter",
            "Evacuate and recharge AC system",
            "Replace AC clutch and pulley",

            # Carburant
            "Replace fuel filter and clean tank",
            "Replace fuel pump and strainer",
            "Clean fuel injectors",
            "Replace fuel pressure regulator",
            "Replace fuel tank vent valve",
            "Replace carbon canister",
            "Replace fuel cap",
            "Repair fuel leak",

            # Échappement
            "Replace exhaust manifold gasket",
            "Replace catalytic converter and oxygen sensors",
            "Replace muffler and tailpipe",
            "Replace exhaust pipe section",
            "Replace heat shield",
            "Repair exhaust leak",

            # Refroidissement
            "Replace radiator and hoses",
            "Replace water pump and thermostat",
            "Replace cooling fan motor",
            "Replace coolant temperature sensor",
            "Flush cooling system and refill",
            "Replace radiator cap",
            "Repair coolant leak",

            # Direction
            "Replace power steering pump and fluid",
            "Replace steering rack and tie rods",
            "Replace power steering hoses",
            "Replace steering column",
            "Flush power steering fluid",
            "Replace steering wheel position sensor",

            # Éclairage
            "Replace headlight bulbs",
            "Replace tail light assembly",
            "Replace turn signal bulbs",
            "Replace brake light switch",
            "Replace headlight switch",
            "Repair lighting circuit",

            # Carrosserie
            "Replace door latch mechanism",
            "Replace window regulator motor",
            "Replace power window motor",
            "Replace door lock actuator",
            "Replace sunroof motor and cables",
            "Replace trunk release actuator",
            "Replace mirror motor",
            "Replace seat motor",

            # Capteurs
            "Replace knock sensor",
            "Replace throttle position sensor",
            "Replace manifold absolute pressure sensor",
            "Replace intake air temperature sensor",
            "Replace coolant temperature sensor",
            "Replace oil pressure sensor",
            "Replace speed sensor",
            "Recalibrate sensor",
            "Clean sensor and connector",

            # Hybride
            "Replace faulty battery modules and rebalance the hybrid battery pack",
            "Replace inverter cooling pump and refill coolant",
            "Update ECU software and reset hybrid system",
            "Replace high voltage cables",
            "Replace DC-DC converter",
            "Replace electric motor bearings",
            "Recalibrate hybrid system",

            # Maintenance préventive
            "Perform scheduled maintenance",
            "Change engine oil and filter",
            "Replace air filter",
            "Replace cabin filter",
            "Rotate tires and check pressure",
            "Adjust tire pressure to specification",
            "Inspect and adjust belts",
            "Check and top off fluids",
            "Perform multi-point inspection",

            # Nettoyage et calibration
            "Clean throttle body and idle air control valve",
            "Clean mass airflow sensor",
            "Clean EGR valve and passages",
            "Clean PCV valve and hoses",
            "Clean battery terminals and apply protectant",
            "Clean and adjust brakes",
            "Clean fuel system with additives",
            "Clean air intake system",
            "Recalibrate steering angle sensor",
            "Recalibrate lane keeping assist camera",
            "Reset adaptive learning values",
            "Clear fault codes and road test",

            # Réparations spécialisées
            "Increase MG1 nut torque spec from 353Nm to 398Nm",
            "Update software and perform relearn procedure",
            "Replace GPS antenna and update maps",
            "Clean radar sensor and recalibrate",
            "Clean parking sensor and test operation",
            "Clean blind spot sensor and verify function",
            "Clean rearview camera lens and adjust",
            "Replace horn relay and test operation",
            "Replace wiper fuse and check motor",
            "Replace heated seat element and test"
        ]

    def load_original_data(self, file_path: str) -> pd.DataFrame:
        """
        Charge les données originales et supprime la colonne Statut GIM

        Args:
            file_path (str): Chemin vers le fichier CSV original

        Returns:
            pd.DataFrame: DataFrame nettoyé sans la colonne Statut GIM
        """
        print(f"Chargement des données originales depuis {file_path}...")
        df = pd.read_csv(file_path)

        print(f"Données originales: {len(df)} lignes, {len(df.columns)} colonnes")
        print(f"Colonnes: {list(df.columns)}")

        # Suppression de la colonne "Statut GIM"
        if 'Statut GIM' in df.columns:
            df = df.drop('Statut GIM', axis=1)
            print("✅ Colonne 'Statut GIM' supprimée")

        print(f"Données après nettoyage: {len(df)} lignes, {len(df.columns)} colonnes")
        return df

    def generate_synthetic_data(self, target_size: int = 5000) -> pd.DataFrame:
        """
        Génère des données synthétiques réalistes

        Args:
            target_size (int): Nombre total d'exemples à générer

        Returns:
            pd.DataFrame: DataFrame avec les données synthétiques
        """
        print(f"Génération de {target_size} exemples synthétiques...")

        synthetic_data = []

        for i in range(target_size):
            # Sélection aléatoire des éléments
            dtc_code = random.choice(self.dtc_codes)
            problem_desc = random.choice(self.problem_descriptions)
            root_cause = random.choice(self.root_causes)
            component = random.choice(self.components)
            pca_solution = random.choice(self.pca_solutions)

            # Ajout de variations dans les descriptions
            problem_desc = self._add_variation_to_description(problem_desc)
            root_cause = self._add_variation_to_root_cause(root_cause)

            synthetic_data.append({
                'ID': i + 1,
                'Code DTC': dtc_code,
                'Description du problème': problem_desc,
                'Root Cause Description': root_cause,
                'Composant concerné': component,
                'PCA attendue': pca_solution
            })

        df_synthetic = pd.DataFrame(synthetic_data)
        print(f"✅ {len(df_synthetic)} exemples synthétiques générés")

        return df_synthetic

    def _add_variation_to_description(self, description: str) -> str:
        """
        Ajoute des variations aux descriptions pour plus de réalisme

        Args:
            description (str): Description originale

        Returns:
            str: Description avec variations
        """
        variations = [
            description,
            description + ", intermittent issue",
            description + ", occurs during cold start",
            description + ", happens when accelerating",
            description + ", noticeable at highway speeds",
            description + ", worse in cold weather",
            description + ", getting progressively worse",
            description + ", started recently",
            description + ", customer reports",
            description + ", technician observed",
            "Intermittent " + description.lower(),
            "Occasional " + description.lower(),
            "Frequent " + description.lower(),
            "Severe " + description.lower(),
            "Mild " + description.lower()
        ]

        return random.choice(variations)

    def _add_variation_to_root_cause(self, root_cause: str) -> str:
        """
        Ajoute des variations aux causes racines

        Args:
            root_cause (str): Cause racine originale

        Returns:
            str: Cause racine avec variations
        """
        variations = [
            root_cause,
            root_cause + " due to age",
            root_cause + " from normal wear",
            root_cause + " caused by contamination",
            root_cause + " resulting from overuse",
            root_cause + " due to environmental factors",
            "Suspected " + root_cause.lower(),
            "Confirmed " + root_cause.lower(),
            "Likely " + root_cause.lower(),
            "Possible " + root_cause.lower()
        ]

        return random.choice(variations)

    def balance_dataset(self, df: pd.DataFrame, min_samples_per_class: int = 50) -> pd.DataFrame:
        """
        Équilibre le dataset en s'assurant qu'il y a assez d'exemples par classe PCA

        Args:
            df (pd.DataFrame): DataFrame à équilibrer
            min_samples_per_class (int): Nombre minimum d'exemples par classe

        Returns:
            pd.DataFrame: DataFrame équilibré
        """
        print("Équilibrage du dataset...")

        # Comptage des classes
        class_counts = df['PCA attendue'].value_counts()
        print(f"Classes avant équilibrage: {len(class_counts)}")
        print(f"Exemples par classe (min/max): {class_counts.min()}/{class_counts.max()}")

        balanced_data = []

        for pca_class in class_counts.index:
            class_data = df[df['PCA attendue'] == pca_class]
            current_count = len(class_data)

            if current_count < min_samples_per_class:
                # Générer plus d'exemples pour cette classe
                needed = min_samples_per_class - current_count

                for _ in range(needed):
                    # Prendre un exemple existant comme base
                    base_example = class_data.sample(1).iloc[0].to_dict()

                    # Modifier légèrement les descriptions
                    base_example['Description du problème'] = self._add_variation_to_description(
                        base_example['Description du problème']
                    )
                    base_example['Root Cause Description'] = self._add_variation_to_root_cause(
                        base_example['Root Cause Description']
                    )

                    # Nouveau ID
                    base_example['ID'] = len(df) + len(balanced_data) + 1

                    balanced_data.append(base_example)

            # Ajouter les exemples existants
            balanced_data.extend(class_data.to_dict('records'))

        df_balanced = pd.DataFrame(balanced_data)

        # Mélanger les données
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        # Réassigner les IDs
        df_balanced['ID'] = range(1, len(df_balanced) + 1)

        # Statistiques finales
        final_class_counts = df_balanced['PCA attendue'].value_counts()
        print(f"Classes après équilibrage: {len(final_class_counts)}")
        print(f"Exemples par classe (min/max): {final_class_counts.min()}/{final_class_counts.max()}")
        print(f"Total d'exemples: {len(df_balanced)}")

        return df_balanced

    def augment_dataset(self, input_file: str, output_file: str,
                       target_size: int = 5000, min_samples_per_class: int = 50) -> None:
        """
        Pipeline complet d'augmentation du dataset

        Args:
            input_file (str): Fichier d'entrée
            output_file (str): Fichier de sortie
            target_size (int): Taille cible du dataset
            min_samples_per_class (int): Minimum d'exemples par classe
        """
        print("=" * 60)
        print("🚀 DÉBUT DE L'AUGMENTATION DU DATASET")
        print("=" * 60)

        # 1. Charger les données originales
        df_original = self.load_original_data(input_file)

        # 2. Générer des données synthétiques
        df_synthetic = self.generate_synthetic_data(target_size - len(df_original))

        # 3. Combiner les données
        print("\nCombinaison des données originales et synthétiques...")
        df_combined = pd.concat([df_original, df_synthetic], ignore_index=True)
        print(f"Dataset combiné: {len(df_combined)} exemples")

        # 4. Équilibrer le dataset
        df_final = self.balance_dataset(df_combined, min_samples_per_class)

        # 5. Sauvegarder
        print(f"\nSauvegarde dans {output_file}...")
        df_final.to_csv(output_file, index=False)

        print("=" * 60)
        print("✅ AUGMENTATION TERMINÉE AVEC SUCCÈS!")
        print("=" * 60)
        print(f"📊 Dataset final: {len(df_final)} exemples")
        print(f"📊 Classes uniques: {df_final['PCA attendue'].nunique()}")
        print(f"📁 Fichier sauvegardé: {output_file}")

        # Affichage des statistiques
        print("\n📈 STATISTIQUES FINALES:")
        class_stats = df_final['PCA attendue'].value_counts()
        print(f"- Nombre de classes: {len(class_stats)}")
        print(f"- Exemples par classe (min/max/moyenne): {class_stats.min()}/{class_stats.max()}/{class_stats.mean():.1f}")
        print(f"- Écart-type: {class_stats.std():.1f}")


def main():
    """Fonction principale"""
    augmenter = DatasetAugmenter()

    # Configuration
    input_file = 'data/gim_diagnostic_dataset.csv'
    output_file = 'data/gim_diagnostic_dataset_augmented.csv'
    target_size = 5000  # Taille cible du dataset
    min_samples_per_class = 100  # Minimum d'exemples par classe

    try:
        augmenter.augment_dataset(
            input_file=input_file,
            output_file=output_file,
            target_size=target_size,
            min_samples_per_class=min_samples_per_class
        )

        print(f"\n🎉 Dataset augmenté disponible dans: {output_file}")
        print("💡 Vous pouvez maintenant réentraîner le modèle avec:")
        print(f"   python main.py --action train --data {output_file}")

    except Exception as e:
        print(f"❌ Erreur lors de l'augmentation: {e}")
        raise


if __name__ == "__main__":
    main()