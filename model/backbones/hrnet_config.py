from easydict import EasyDict

hrnet_config = EasyDict()

# width × height
hrnet_config.IMAGE_SIZE = [192, 256]
hrnet_config.HEATMAP_SIZE = [48, 64]
hrnet_config.POST_PROCESS = True
hrnet_config.ADDITIONAL_LAYER = False

hrnet_config.NUM_JOINTS = 17
hrnet_config.PRETRAINED_LAYERS = ['*']
hrnet_config.STEM_INPLANES = 64
hrnet_config.FINAL_CONV_KERNEL = 1

hrnet_config.STAGE2 = EasyDict()
hrnet_config.STAGE2.NUM_MODULES = 1
hrnet_config.STAGE2.NUM_BRANCHES = 2
hrnet_config.STAGE2.NUM_BLOCKS = [4, 4]
hrnet_config.STAGE2.NUM_CHANNELS = [32, 64]
# hrnet_config.STAGE2.NUM_CHANNELS = [48, 96]
hrnet_config.STAGE2.BLOCK = 'BASIC'
hrnet_config.STAGE2.FUSE_METHOD = 'SUM'

hrnet_config.STAGE3 = EasyDict()
# hrnet_config.STAGE3.NUM_MODULES = 1
hrnet_config.STAGE3.NUM_MODULES = 4
hrnet_config.STAGE3.NUM_BRANCHES = 3
hrnet_config.STAGE3.NUM_BLOCKS = [4, 4, 4]
hrnet_config.STAGE3.NUM_CHANNELS = [32, 64, 128]
# hrnet_config.STAGE3.NUM_CHANNELS = [48, 96, 192]
hrnet_config.STAGE3.BLOCK = 'BASIC'
hrnet_config.STAGE3.FUSE_METHOD = 'SUM'

hrnet_config.STAGE4 = EasyDict()
# hrnet_config.STAGE4.NUM_MODULES = 1
hrnet_config.STAGE4.NUM_MODULES = 3
hrnet_config.STAGE4.NUM_BRANCHES = 4
hrnet_config.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
hrnet_config.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
# hrnet_config.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
hrnet_config.STAGE4.BLOCK = 'BASIC'
hrnet_config.STAGE4.FUSE_METHOD = 'SUM'
