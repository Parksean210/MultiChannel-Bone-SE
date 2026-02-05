import lightning as L
from lightning.pytorch.callbacks import Callback
import pynvml
import logging

log = logging.getLogger(__name__)

class GPUStatsMonitor(Callback):
    """
    모든 배치가 끝날 때마다 GPU 상태(사용률, 메모리, 온도)를 로깅하는 커스텀 콜백.
    Lightning의 DeviceStatsMonitor가 GPU 사용률을 제공하지 않아 pynvml을 직접 사용함.
    """
    def __init__(self, device_index=0):
        super().__init__()
        import os
        self.device_index = device_index
        try:
            pynvml.nvmlInit()
            
            # 슈퍼컴퓨터(SLURM 등) 환경 대응: CUDA_VISIBLE_DEVICES 기반 실제 물리 인덱스 찾기
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible_devices:
                # 예: "2,3" 이면 logical 0은 물리 2번임
                physical_indices = [int(x.strip()) for x in visible_devices.split(",")]
                if device_index < len(physical_indices):
                    physical_idx = physical_indices[device_index]
                    log.info(f"Mapping logical GPU {device_index} to physical GPU {physical_idx} (from CUDA_VISIBLE_DEVICES)")
                else:
                    physical_idx = device_index
            else:
                physical_idx = device_index
                
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)
            self.physical_idx = physical_idx
            self._available = True
        except Exception as e:
            log.warning(f"Failed to initialize pynvml: {e}. GPU monitoring disabled.")
            self._available = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._available:
            return
            
        # Standardize keys to match MLFlow System Metrics format
        # This forces the files to be saved in metrics/system/ directory,
        # which MLFlow UI picks up for the "System Metrics" tab.
        
        # 1. GPU Utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        pl_module.log(f"system/gpu_{self.device_index}_utilization_percentage", float(util.gpu), prog_bar=True, logger=True)
        
        # 2. GPU Memory Usage
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        pl_module.log(f"system/gpu_{self.device_index}_memory_usage_megabytes", mem.used / 1024**2, prog_bar=False, logger=True)
        pl_module.log(f"system/gpu_{self.device_index}_memory_usage_percentage", (mem.used / mem.total) * 100, prog_bar=False, logger=True)
        
        # 3. Temperature
        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        pl_module.log(f"system/gpu_{self.device_index}_temperature_celsius", float(temp), prog_bar=False, logger=True)

    def teardown(self, trainer, pl_module, stage: str):
        if self._available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
