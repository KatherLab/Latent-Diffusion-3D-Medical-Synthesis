def create_model():
    
    from .PTN_model3D import PTN_local_trans
    from .networks import define_D_3D as define_D
    from .pre_r3d_18 import Res3D

    model = PTN_local_trans(img_size=patch_size)
    # ext_discriminator = Res3D()
    # D = define_D(2, 64, 2, 'instance3D', False, 3, True, [0])
    
    return model

   