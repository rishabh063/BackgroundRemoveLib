import urllib.request
import os 
import torch
class BackGroundProcessor:
    device='cuda'
    base_model_link='https://ecombuckets3.s3.ap-south-1.amazonaws.com/BackGroundRem/inspyrenetBase.pth'
    base_ckpt_dir='model/'
    base_ckpt_name='backgroundRemBase.pth'
    base_size=[1024,1024]
    def DownLoadBaseModel(self):
        urllib.request.urlretrieve(self.base_model_link, self.base_ckpt_dir + self.base_ckpt_name)
    
    def check_models(self):
        if os.path.exists(self.base_ckpt_dir + self.base_ckpt_name):
            pass
        else:
            self.DownLoadBaseModel()
    def loadModel(self):
        ckpt_name = self.base_ckpt_name.replace(".pth", "_{}.pt".format(self.device))
        try:
            traced_model = torch.jit.load(os.path.join(self.base_ckpt_dir, ckpt_name), map_location=self.device)
            self.model = traced_model
        except:
            self.check_models()
            self.model = InSPyReNet_SwinB(depth=64, pretrained=False, threshold=None)
            self.model.eval()
            self.model.load_state_dict(torch.load(os.path.join(self.base_ckpt_dir  , self.base_ckpt_name), map_location="cpu"),strict=True,)
            self.model = self.model.to(self.device)
            traced_model = torch.jit.trace(self.model,torch.rand(1, 3, self.base_size).to(self.device),strict=True)
            del self.model
            self.model = traced_model
            torch.jit.save(self.model, os.path.join(self.base_ckpt_dir, ckpt_name))
        self.transform = transforms.Compose([
            transforms.Resize(base_size),  # Resize image to base_size
            transforms.ToTensor(),         # Convert image to a PyTorch Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
            ])

    def Proccess(self , img):
        shape = img.size[::-1]
        x = self.transform(img)
        x = x.unsqueeze(0)
        x = x.to(self.device)
        with torch.no_grad():
            pred = self.model(x)
        pred = F.interpolate(pred, shape, mode="bilinear", align_corners=True)
        pred = pred.data.cpu()
        pred = pred.numpy().squeeze()

        if threshold is not None:
            pred = (pred > float(threshold)).astype(np.float64)

        img = np.array(img)
        r, g, b = cv2.split(img)
        pred = (pred * 255).astype(np.uint8)
        img = cv2.merge([r, g, b, pred])
        return Image.fromarray(img.astype(np.uint8))
