# TVN
**[2020/5/8]** This repo is cloned from https://github.com/DELTA37/TVN. However, some codes need to be repaired, so I modified these codes for GPU-training and resume training.

**[TODO] Speed up dataloader** 

------

**Tiny Video Networks**

![TVN architecture](/static/tvn.png)

pip install tvn==1.0 


After installing or cloning  
>> import tvn  
>> from tvn.data import SomethingSomethingV2  
>> from tvn.config import CFG1  
>> from tvn.model import TVN  
>> from tvn.solver import Solver  


You should download data from [SomethingSomethingV2 dataset](https://20bn.com/datasets/something-something)  
And place it in ./data folder  
After that you can start training via script train.py  
>> python train.py  

Pretrained models will be as soon as issue with this question appears :)  

You can specify your own config in the way as in tvn.config  

If you find any bugs please contact me..  