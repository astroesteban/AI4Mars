# import timm

# def calc_size(model_name: str) -> int:
#     # create a pyotrch model from the name
#     model = timm.create_model(model_name)

#     return calc_model_size(model, False)

# model_list = pd.DataFrame({"Name": timm.list_models('efficientnet*', pretrained=True)})
# model_list["Size (MB)"] = model_list.Name.apply(calc_size)

# model_list


# msk_t = torchvision.transforms.functional.pil_to_tensor(msk)
# msk_t[msk_t == 255] = 4

# msk_t
# cleanup_gpu_cache()

# model_architecture = "efficientnet_lite0.ra_in1k"

# # Create a pretrained ResNet model with the number of output classes equal to the number of class names
# # 'timm.create_model' function automatically downloads and initializes the pretrained weights
# efficientnet = timm.create_model(model_architecture)

# # Set the device and data type for the model
# efficientnet = efficientnet.to(device=g_DEVICE, dtype=torch.float32)

# # Add attributes to store the device and model name for later reference
# efficientnet.device = g_DEVICE
# resnet18.name = model_architecture

# my_net = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large()

# fastai_nnet = Learner(dls=ai4mars_data_loader, model=my_net)

# learn = unet_learner(ai4mars_data_loader, resnet18)
