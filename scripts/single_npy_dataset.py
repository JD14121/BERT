import numpy as np,torch
from pathlib import Path
from torch.utils.data import Dataset

class SingleNpyDataset(Dataset):
    def __init__(self,npy_dir,distance=7,rounds=10,p=0.0,snr=10.0):
        self.distance=distance;self.rounds=rounds;self.p=p;self.snr=snr
        self.n_stab=distance*distance-1;self.n_data=distance*distance
        num_det=rounds*self.n_stab
        d=Path(npy_dir)
        N=(d/"label.npy").stat().st_size//4
        self.num_samples=N
        self._m={
            "measurement":np.memmap(str(d/"measurement.npy"),dtype=np.float32,mode="r",shape=(N,rounds,self.n_stab)),
            "event":np.memmap(str(d/"event.npy"),dtype=np.float32,mode="r",shape=(N,rounds,self.n_stab)),
            "final_soft":np.memmap(str(d/"final_soft.npy"),dtype=np.float32,mode="r",shape=(N,self.n_data)),
            "label":np.memmap(str(d/"label.npy"),dtype=np.float32,mode="r",shape=(N,)),
            "detection_events":np.memmap(str(d/"detection_events.npy"),dtype=np.float32,mode="r",shape=(N,num_det)),
        }
        from alphaqubit.data.coordinates import CoordinateSystem
        self._coord_system=CoordinateSystem(distance)
    @property
    def coord_system(self): return self._coord_system
    def __len__(self): return self.num_samples
    def __getitem__(self,idx):
        T=self.rounds
        return {"measurement":torch.from_numpy(np.array(self._m["measurement"][idx])),
                "event":torch.from_numpy(np.array(self._m["event"][idx])),
                "final_soft":torch.from_numpy(np.array(self._m["final_soft"][idx])),
                "label":torch.tensor([float(self._m["label"][idx])],dtype=torch.float32),
                "leakage":torch.zeros(T,self.n_stab,dtype=torch.float32),
                "event_leakage":torch.zeros(T,self.n_stab,dtype=torch.float32),
                "stab_pos_idx":self._coord_system.scatter_idx.clone()}

def load_single_npy(d,r,basis,data_dir):
    return SingleNpyDataset(Path(data_dir)/f"d{d}"/"npy_large",distance=d,rounds=r)
