import numpy as np
import cv2
class Filt:
    def __init__(self, path):
        self.path = path
        self.filtro = None
    def label(self):
        filename = self.path.split('\\')[-1]  
        filtro_part = filename.split('-')[0] 
        self.filtro = filtro_part
        if filtro_part=="filtro_1":
            self.filtro = filtro_part
            return 1           
        elif filtro_part=="filtro_2" : 
            self.filtro = filtro_part 
            return 2
        elif filtro_part=="filtro_3" :  
            self.filtro = filtro_part
            return 3
        elif filtro_part=="filtro_4":  
            self.filtro = filtro_part
            return 4
    def corde(self):
        cordl=[]
        cordr=[]
        if self.filtro=="filtro_1":
            cordl=np.array([[342,318],[329,348],[345,355],[358,325]], dtype=np.int32)
            cordr=np.array([[556,398],[544,430],[560,435],[570,404]], dtype=np.int32)
            return (cordl,cordr)           
        elif self.filtro=="filtro_2": 
            cordl=np.array([[511,217],[512,253],[522,253],[523,216]], dtype=np.int32)
            cordr=np.array([[720,218],[721,258],[735,258],[735,217]], dtype=np.int32)
            return (cordl,cordr)
        elif self.filtro=="filtro_3":  
            cordl=np.array([[459,301],[459,339],[473,340],[472,301]], dtype=np.int32)
            cordr=np.array([[672,298],[673,338],[690,339],[689,298]], dtype=np.int32)
            return (cordl,cordr)
        elif self.filtro=="filtro_4":  
            cordl=np.array([[472,309],[468,346],[482,345],[485,310]], dtype=np.int32) 
            cordr=np.array([[687,319],[688,360],[704,362],[703,319]], dtype=np.int32)
            return (cordl,cordr)
    def umbral(self):
        umb=[0,0]
        if self.filtro=="filtro_1":
            umb[0]=50
            umb[1]=45
            return umb           
        elif self.filtro=="filtro_2": 
            umb[0]=50
            umb[1]=45
            return umb
        elif self.filtro=="filtro_3": 
            umb[0]=50
            umb[1]=45
            return umb
        elif self.filtro=="filtro_4": 
            umb[0]=50
            umb[1]=45
            return umb
        
    def adjust(self):
        adj=[0,0,0,0]
        if self.filtro=="filtro_1":
            adj[0]=140
            adj[1]=2.5
            adj[2]=127
            adj[3]=2.1
            return adj           
        elif self.filtro=="filtro_2": 
            adj[0]=140
            adj[1]=2.5
            adj[2]=127
            adj[3]=2.1
            return adj
        elif self.filtro=="filtro_3": 
            adj[0]=140
            adj[1]=2.5
            adj[2]=127
            adj[3]=2.1
            return adj
        elif self.filtro=="filtro_4": 
            adj[0]=140
            adj[1]=2.5
            adj[2]=127
            adj[3]=2.1
            return adj
    def mask(self):
        mask1=[]
        mask2=[]
        mask3=[]
        if self.filtro=="filtro_1":
            mask1=np.array([[668,243],[624,382],[602,373],[568,471],[536,459],[508,540],[677,540],[764,282]], dtype=np.int32)
            mask2=np.array([[0,0],[0,121],[181,0]], dtype=np.int32)
            mask3=np.array([[608,320],[597,349],[577,345],[574,350],[604,359],[618,323]], dtype=np.int32)
            return (mask1,mask2,mask3)           
        elif self.filtro=="filtro_2": 
            mask1=np.array([[790,48],[801,248],[882,233],[868,250]], dtype=np.int32)
            mask2=np.array([[0,109],[0,290],[62,287],[56,135]], dtype=np.int32)
            mask3=np.array([[730,169],[737,185],[754,186],[760,170]], dtype=np.int32)
            return (mask1,mask2,mask3)
        elif self.filtro=="filtro_3": 
            mask1=np.array([[741,133],[754,344],[813,340],[798,133]], dtype=np.int32)
            mask2=np.array([[0,126],[0,370],[104,319],[106,211]], dtype=np.int32)
            mask3=np.array([[693,245],[696,263],[725,260],[728,246]], dtype=np.int32)
            return (mask1,mask2,mask3)
        elif self.filtro=="filtro_4": 
            mask1=np.array([[760,73],[769,393],[830,390],[826,219],[808,74]], dtype=np.int32)
            mask2=np.array([[0,94],[1,340],[65,304],[71,178]], dtype=np.int32)
            mask3=np.array([[712,267],[716,282],[744,284],[749,269]], dtype=np.int32)
            return (mask1,mask2,mask3)
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return limg

#filt_instance = Filt(r"C:\repos\antamina-filter-clogging-preventer\datasets\antamina-filter-segmentation.v4i.yolov8\test\images\filtro_4-2024-01-30_11-54-11_png.rf.cf1fe7c620c35888ea35af3adf6c8d11.jpg")
#print(filt_instance.label())
