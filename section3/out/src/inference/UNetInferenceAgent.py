"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        
        # Working through this: this scenario is the same as running `single_volume_inference`, but we have to
        # pad it first (with the `med_reshape` function. 
        
        # First, reshape. Previous implementations look like this:
        # med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape))
        
        # I thought we could pull this. For now, hard code
        patch_size = 64 
        
        print("New shape ", volume.shape[0], patch_size, patch_size)
        reshaped_volume = med_reshape(volume, new_shape=(volume.shape[0], patch_size, patch_size))
        
        # Now we run the `singled_volume_inference` function on this reshaped volume
        # We have precedent for this too:
        # inference_agent.single_volume_inference(x["image"])
        
        # I think we can run direclty on this reshaped_volume...
        return self.single_volume_inference(reshaped_volume)
        

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = np.zeros(volume.shape)

        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        # <YOUR CODE HERE>
        
        for slice_position in range(volume.shape[0]):
            
            # Note that this is different than the min max scaler used previously... 
            # I freely acknowledge that this is not ideal (but comparable). 
            norm_slice = volume[slice_position, :, :].astype(np.single) / 255.0
            
            pred = self.model(torch.from_numpy(norm_slice).unsqueeze(0).unsqueeze(0).to(self.device))
            pred_values = np.squeeze(pred.cpu().detach())
            
            slices[slice_position, :, :] = torch.argmax(pred_values, dim=0)
        
        return slices 
