from .cascadepspRun import CascadePSP
from .removeBackgroundpipe import BackGroundProcessor
remove=BackGroundProcessor()
remove.loadModel()
casade=CascadePSP(device='cuda' , fp16=True)


def RemoveBackground(inputimg):
    output=remove.Proccess(inputimg.convert('RGB'))
    inputimg.putalpha(casade([inputimg.convert('RGB')],[output.split()[3]])[0])
    return inputimg