import os 
import urllib
url={
    'fba_matting.pth':'https://ecombuckets3.s3.ap-south-1.amazonaws.com/BackGroundRem/fba_matting.pth',
    'cascadepsp_finetuned_carveset.pth':'https://ecombuckets3.s3.ap-south-1.amazonaws.com/BackGroundRem/cascadepsp_finetuned_carveset.pth'
}
checkpoints_dir='models/'
def downloader(name):
    if os.path.exists(checkpoints_dir + name):
        return checkpoints_dir + name
    else:
        os.makedirs(checkpoints_dir, exist_ok=True)
        urllib.request.urlretrieve(url[name], checkpoints_dir + name)
    return checkpoints_dir + name


