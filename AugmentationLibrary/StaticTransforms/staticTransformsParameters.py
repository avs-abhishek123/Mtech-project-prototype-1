Static_transforms_dict=dict()
Static_transforms_dict={
    'Blur':
    {
        "blur_limit": {"data_type":"float","min": 3,"max":"infinity", "default":7},
        "always_apply":{"data_type":"Boolean", "default":False},
        "probaility":{"data_type":"float", "default":1.0}
    },
    
    'VerticalFlip':
    {
        "probaility":{"data_type":"float", "default":1.0}
    },

    'HorizontalFlip':
    {
        "probaility":{"data_type":"float", "default":1.0}    
    },
    
    'Flip':
    {
        "probaility":{"data_type":"float", "default":1.0}
    },
    
    'Normalize':
    {
        # Normalization is applied by the formula:
        # img = (img - mean * max_pixel_value) / (std * max_pixel_value)

        "mean":{"data_type":"(tuple of float)", "default":(0.485, 0.456, 0.406)},
        "std":{"data_type":"(tuple of float)", "default":(0.229, 0.224, 0.225)},
        "max_pixel_value":{"data_type":"(tuple of float)", "default":255.0},
        "always_apply":{"data_type":"Boolean", "default":False},
        "probaility":{"data_type":"float", "default":1.0}
    },

    'Transpose':
    {
        "probaility":{"data_type":"float", "default":1.0}
    },
    'OpticalDistortion':
    {
        "distort_limit":{"data_type":"(float, (float, float))", "default":(-0.05,0.05)}, 
        "shift_limit":{"data_type":"(float, (float, float))", "default":(-0.05,0.05)}, 
        # interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
        # cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
        "interpolation":{"data_type":["cv2.INTER_NEAREST", "cv2.INTER_LINEAR", "cv2.INTER_CUBIC", "cv2.INTER_AREA", "cv2.INTER_LANCZOS4"], "default":{1:"cv2.INTER_LINEAR"}}, 
        # border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
        # cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
        "border_mode":{"data_type":["cv2.BORDER_CONSTANT", "cv2.BORDER_REPLICATE", "cv2.BORDER_REFLECT", "cv2.BORDER_WRAP", "cv2.BORDER_REFLECT_101"], "default":{4:"cv2.BORDER_REFLECT_101"}}, 
        "value":{"data_type":"int, float, list of ints,list of float", "default":None},
        "mask_value":{"data_type":"int, float, list of ints,list of float", "default":None},
        "always_apply":{"data_type":"Boolean", "default":False},
        "probaility":{"data_type":"float", "default":1.0}
    },
    'GridDistortion':
    {
        #num_steps (int): count of grid cells on each side.
        "num_steps":{"data_type":"int", "default":5},
        "distort_limit":{"data_type":"(float, (float, float))", "default":(-0.03,0.03)}, 
        # interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
        # cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
        "interpolation":{"data_type":["cv2.INTER_NEAREST", "cv2.INTER_LINEAR", "cv2.INTER_CUBIC", "cv2.INTER_AREA", "cv2.INTER_LANCZOS4"], "default":{1:"cv2.INTER_LINEAR"}}, 
        # border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
        # cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
        "border_mode":{"data_type":["cv2.BORDER_CONSTANT", "cv2.BORDER_REPLICATE", "cv2.BORDER_REFLECT", "cv2.BORDER_WRAP", "cv2.BORDER_REFLECT_101"], "default":{4:"cv2.BORDER_REFLECT_101"}}, 
        # value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        "value":{"data_type":"int, float, list of ints,list of float", "default":None},
        "mask_value":{"data_type":"int, float, list of ints,list of float", "default":None},
        "always_apply":{"data_type":"Boolean", "default":False},
        "probaility":{"data_type":"float", "default":1.0}    
    },
    'HueSaturationValue':
    {
        "hue_shift_limit" : {"data_type":"(int, int) or int", "default":(-20, 20)},
        "sat_shift_limit" : {"data_type":"(int, int) or int", "default":(-30, 30)},
        "val_shift_limit" : {"data_type":"(int, int) or int", "default":(-20, 20)},
        "always_apply":{"data_type":"Boolean", "default":False},
        "probaility":{"data_type":"float", "default":1.0}        
         
    },
    """
    'PadIfNeeded' :
    {
        "min_height":{"data_type":"int", "default":1024}, 
        "min_width":{"data_type":"int", "default":1024}, 
        "pad_height_divisor":{"data_type":"int", "default":None}, 
        "pad_width_divisor":{"data_type":"int", "default":None},         
        # position (Union[str, PositionType]): Position of the image. should be 
        # PositionType.CENTER or PositionType.TOP_LEFT or PositionType.TOP_RIGHT or PositionType.BOTTOM_LEFT or PositionType.BOTTOM_RIGHT.
        # Original Documentation Default: PositionType.CENTER
        # Default : None                
        "position":{"data_type":[ "PositionType.CENTER", "PositionType.TOP_LEFT", "PositionType.TOP_RIGHT", "PositionType.BOTTOM_LEFT" ,"PositionType.BOTTOM_RIGHT"], "default":"PositionType.CENTER"}, 
        "border_mode":{"data_type":["cv2.BORDER_CONSTANT", "cv2.BORDER_REPLICATE", "cv2.BORDER_REFLECT", "cv2.BORDER_WRAP", "cv2.BORDER_REFLECT_101"], "default":{4:"cv2.BORDER_REFLECT_101"}}, 
        "value":{"data_type":"int, float, list of ints,list of float", "default":None},        
        "mask_value":{"data_type":"int, float, list of ints,list of float", "default":None},
        "always_apply":{"data_type":"Boolean", "default":False},
        "probaility":{"data_type":"float", "default":1.0}  
    },"""
    'RGBShift' :
    {        
        "r_shift_limit" : {"data_type":"(int, int) or int", "default":(-20, 20)},
        "g_shift_limit" : {"data_type":"(int, int) or int", "default":(-20, 20)},             
        "b_shift_limit" : {"data_type":"(int, int) or int", "default":(-20, 20)},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}       
    },
    'MotionBlur' :
    {
        "blur_limit" : {"data_type":"int, (int, int)", "default":(3,7)},       
        "probaility" : {"data_type":"float", "default":1.0}       
    },
    'MedianBlur':
    {
        "blur_limit" : {"data_type":"int, (int, int)", "default":(3,7)},       
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}            
    },
    'GaussianBlur':
    {
        "blur_limit" : {"data_type":"int, (int, int)", "default":(3,7)},  
        # sigma_limit (float, (float, float)): Gaussian kernel standard deviation. Must be greater in range [0, inf).
        # If set single value `sigma_limit` will be in range (0, sigma_limit).
        # If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.        
        "sigma_limit" : {"data_type":"float, (float, float)", "default":(0,0)},     
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}           
    },
    'GaussNoise':
    {
        "var_limit": {"data_type":"(float, float) or float", "default":(10.0, 50.0)},
        "mean": {"data_type":"float", "default":0},
        "per_channel": {"data_type":"Boolean", "default":True},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}   
    },
    'GlassBlur':
    {
        "sigma" : {"data_type":"float", "default":0.7},
        "max_delta" : {"data_type":"int", "default":4},
        "iterations" : {"data_type":"int", "default":2},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "mode" : {"data_type":"str", "default":'fast'},
        "probaility" : {"data_type":"float", "default":1.0}   
    },
    'CLAHE':
    {
        "clip_limit" : {"data_type":"float or (float, float)", "default":(1,4)},
        "tile_grid_size" : {"data_type":"(int, int)", "default":(8, 8)},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}   
    },
    'ChannelShuffle':
    {
        "probaility" : {"data_type":"float", "default":1.0}   
    },
    'InvertImg':
    {
        "probaility" : {"data_type":"float", "default":1.0}            
    },
    'ToGray':
    {
        "probaility" : {"data_type":"float", "default":1.0}                     
    },
    'ToSepia':
    {
        "probaility" : {"data_type":"float", "default":1.0}                    
    },
    'JpegCompression':
    {
        "quality_lower" : {"data_type":"float", "default":99},
        "quality_upper" : {"data_type":"float", "default":100},   
        # compression_type (ImageCompressionType): should be ImageCompressionType.JPEG or ImageCompressionType.WEBP
        "ImageCompressionType" : {"data_type":["ImageCompressionType.JPEG", "ImageCompressionType.WEBP"], "default":"ImageCompressionType.JPEG"},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}            
    },
    """
    'ImageCompression':
    {
        "quality_lower" : {"data_type":"float", "default":99},
        "quality_upper" : {"data_type":"float", "default":100},   
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}            
    },"""
    'Cutout':
    {
        "num_holes" : {"data_type":"int", "default":8},
        "max_h_size" : {"data_type":"int", "default":8}, 
        "max_w_size" : {"data_type":"int", "default":8},
        "fill_value" : {"data_type":"int", "default":0}, 
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0} 
    },
    'CoarseDropout':
    {
        "max_holes" : {"data_type":"int", "default":8},
        "max_height" : {"data_type":"int", "default":8}, 
        "max_width" : {"data_type":"int", "default":8},
        "min_holes" : {"data_type":"int", "default":None}, 
        "min_height" : {"data_type":["int", "float"], "default":None}, 
        "min_width" : {"data_type":["int", "float"], "default":None}, 
        "fill_value" : {"data_type":["int", "float", ["int"], ["float"]], "default":0}, 
        "mask_fill_value" : {"data_type":["int", "float", ["int"], ["float"]], "default":None}, 
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}          
    },
    'ToFloat':
    {
        "max_value" : {"data_type":"float", "default":None},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}              
    },
    'FromFloat':
    {
        "max_value" : {"data_type":"float", "default":None},
        "dtype" : {"data_type":["str", "numpy data type"], "default":"uint16"},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}              
    },
    """
    'Lambda':
    {
        "image" : {"data_type":"Image transformation function", "default":None},
        "mask" : {"data_type":"Mask transformation function", "default":None}, 
        "keypoint" : {"data_type":"Keypoint transformation function", "default":None},
        "bbox" : {"data_type":"BBox transformation function", "default":None},       
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0} 
    },"""
    'ChannelDropout':
    {
        "channel_drop_range" : {"data_type":"(int, int)", "default":(1, 1)},
        "fill_value" : {"data_type":["int", "float"], "default":0},      
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}          
    },
    'ISONoise':
    {
        "color_shift" : {"data_type":"(float, float)", "default":(0.01, 0.05)},
        "intensity" : {"data_type":"(float, float)", "default":(0.1, 0.5)},       
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}          
    },
    'Solarize':
    {      
        "threshold" : {"data_type":["(int, int)", "int", "(float, float)" ,"float"], "default":128},       
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0} 
    },
    """
    'Equalize':
    {
        "mode" : {"data_type":"str (cv,pil)", "default":"cv"},       
        "by_channels" : {"data_type":"Boolean", "default":True},       
        # mask (np.ndarray, callable): If given, only the pixels selected by
        # the mask are included in the analysis. Maybe 1 channel or 3 channel array or callable.
        # Function signature must include `image` argument.        
        "mask" : {"data_type":"(np.ndarray, callable)", "default":None},       
        "mask_params" : {"data_type":"list of str", "default":None},       
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}         
    },"""
    """
    'FDA':
    {
        "reference_images":{"data_type":"numpy.ndarray", "default":None},
        "beta_limit" : {"data_type":"float", "default":0.1},
        "read_fn" : {"data_type":"str", "default":""},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}         

    },"""
    """
    'HistogramMatching':
    {
        "reference_images":{"data_type":"numpy.ndarray", "default":None},
        "blend_ratio" : {"data_type":"(float,float)", "default":(0.5, 1.0)},
        "read_fn" : {"data_type":"str", "default":""},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}         

    },"""
    """
    'PixelDistributionAdaptation':
    {
        "reference_images":{"data_type":"numpy.ndarray", "default":None},
        "blend_ratio" : {"data_type":"(float,float)", "default":(0.25, 1.0)},
        "read_fn" : {"data_type":"str", "default":""},
        "transform_type" : {"data_type":["pca", "standard", "minmax"], "default":"pca"},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}         

    },"""

    'Posterize':
    {
        "num_bits" : {"data_type":["(int, int)", "int", "list of ints [r, g, b]","list of ints [[r1, r1], [g1, g2], [b1, b2]]"], "default":4},       
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}                
    },
    'Downscale':
    {
        "scale_min" : {"data_type":"float", "default":0.25},
        "scale_max" : {"data_type":"float", "default":0.25},  
        "interpolation" : {"data_type": ["cv2.INTER_NEAREST", "cv2.INTER_LINEAR", "cv2.INTER_CUBIC", "cv2.INTER_AREA", "cv2.INTER_LANCZOS4"], "default":{0:"cv2.INTER_NEAREST"}},
        # border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
        # cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101. 
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}                 
    },
    'MultiplicativeNoise':
    {
        "multiplier" : {"data_type":"float or tuple of floats", "default":(0.9, 1.1)},
        "per_channel" : {"data_type":"Boolean", "default":False},
        "elementwise" : {"data_type":"Boolean", "default":False},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}          
    },
    'FancyPCA':
    {
        "alpha" : {"data_type":"float", "default":0.1},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}          
    },
    'MaskDropout':
    {
        "max_objects" : {"data_type":"int", "default":1},
        "image_fill_value" : {"data_type":"int", "default":0},
        "mask_fill_value" : {"data_type":"int", "default":0},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}           
    },
    'GridDropout':
    {
        "ratio" : {"data_type":"int", "default":0.5},
        "unit_size_min" : {"data_type":"int", "default":None},
        "unit_size_max" : {"data_type":"int", "default":None},
        "holes_number_x" : {"data_type":"int", "default":None},
        "holes_number_y" : {"data_type":"int", "default":None},
        "shift_x" : {"data_type":"int", "default":0},
        "shift_y" : {"data_type":"int", "default":0},
        "random_offset" : {"data_type":"int", "default":False},
        "fill_value" : {"data_type":"int", "default":0},
        "mask_fill_value" : {"data_type":"int", "default":None},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}          
    },
    'ColorJitter':
    {
        "brightness" : {"data_type":"int", "default":0.2},
        "contrast" : {"data_type":"int", "default":0.2},
        "saturation" : {"data_type":"int", "default":0.2},
        "hue" : {"data_type":"int", "default":0.2},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}      
    },
    'Sharpen':
    {
        "alpha":{"data_type":"(float, float)", "default":(0.2, 0.5)}, 
        "lightness":{"data_type":"(float, float)", "default":(0.5, 1.0)},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}          
    },
    'Emboss':
    {
        "alpha":{"data_type":"(float, float)", "default":(0.2, 0.5)}, 
        "lightness":{"data_type":"(float, float)", "default":(0.2, 0.7)},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}          
    },

    """
    'TemplateTransform':
    {
        "templates" : {"data_type":"numpy array or list of numpy arrays", "default":None},
        "img_weight" : {"data_type":"(float, float) or float", "default":0.5},
        "template_weight" : {"data_type":"(float, float) or float", "default":0.5},
        # template_transform: transformation object which could be applied to template,
        # must produce template the same size as input image.
        "template_transform" : {"data_type":"Boolean", "default":None},
        "name" : {"data_type":"str", "default":None},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0} 
    },
    'RingingOvershoot':
    {
        "blur_limit" : {"data_type":["int", "(int, int)"],"range":"[3, inf)", "default":(7, 15)},
        "cutoff" : {"data_type":["float", "(float, float)"], "default":(0.7853981633974483, 1.5707963267948966)},
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0} 
    },
    'UnsharpMask':
    {
        "blur_limit" : {"data_type":["int", "(int, int)"],"range":"[3, inf)", "default":(3, 7)},
        "sigma_limit" : {"data_type":["float", "(float, float)"], "default":0.0},
        "alpha" : {"data_type":["float", "(float, float)"], "default":(0.2, 1.0)},
        "threshold" : {"data_type":"int", "range": [0, 255] , "default":10},
        "always_apply" : {"data_type":"Boolean","default":False},
        "probaility" : {"data_type":"float", "default":1.0}          
    }"""
    
        'Superpixels':
    {
        "p_replace" : {"data_type":"float or tuple of float", "default":0.1},
        "n_segments" : {"data_type":"int, or tuple of int", "default":100},
        "max_size" : {"data_type":"int", "default":128},
        # interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
        # cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
        "interpolation":{"data_type":["cv2.INTER_NEAREST", "cv2.INTER_LINEAR", "cv2.INTER_CUBIC", "cv2.INTER_AREA", "cv2.INTER_LANCZOS4"], "default":{1:"cv2.INTER_LINEAR"}}, 
        "always_apply" : {"data_type":"Boolean", "default":False},
        "probaility" : {"data_type":"float", "default":1.0}           
    }
}  