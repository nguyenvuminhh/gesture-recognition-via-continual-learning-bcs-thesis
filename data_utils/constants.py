from enum import StrEnum, auto

class DataSetNameClass(StrEnum):
    BASE = 'base'
    DIFFERENT_PARTICIPANTS = 'different_participants'
    DIST_150CM = '1.5m'
    ROT_15 = '15'
    ROT_30 = '30'
    ROT_45 = '45'
    ROT_NEG_15 = '-15'
    ROT_NEG_30 = '-30'
    ROT_NEG_45 = '-45'
    ANGLE_15_DU = '15_du'
    ANGLE_30_DU = '30_du'
    ANGLE_45_DU = '45_du'
    ANGLE_NEG_15_DU = '-15_du'
    ANGLE_NEG_30_DU = '-30_du'
    ANGLE_NEG_45_DU = '-45_du'
    NEW_CLASSES = 'new_classes'

DATA_SET_NAME_LIST = [member.value for member in DataSetNameClass]
# DATA_SET_NAME_LIST = [DataSetNameClass.DIFFERENT_PARTICIPANTS.value, DataSetNameClass.NEW_CLASSES]

NOF_CONTEXTS = len(DATA_SET_NAME_LIST)


class ContinualLearningMethodClass(StrEnum):
    NONE = 'NONE'
    JOINT = 'JOINT'
    EWC = 'EWC'
    SI = 'SI'
    LwF = 'LwF'
    ER = 'ER'
    AGEM = 'AGEM'

CONTINUAL_LEARNING_METHOD_LIST = [member.value for member in ContinualLearningMethodClass]
