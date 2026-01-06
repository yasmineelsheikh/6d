"""
Compatibility layer for decord video reader.
Falls back to opencv-python when decord is not available (e.g., on macOS).
"""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import decord
    DECORD_AVAILABLE = True
    logger.debug("decord available - using native implementation")
except ImportError:
    DECORD_AVAILABLE = False
    # On Linux (e.g., AWS), decord should be available. On macOS, use fallback.
    import platform
    if platform.system() == "Darwin":
        logger.info("macOS detected: decord not available, using opencv-python fallback")
    else:
        logger.warning("decord not available on Linux. Consider installing: pip install decord")


class VideoReader:
    """
    Compatibility wrapper for decord.VideoReader that falls back to opencv.
    """
    
    def __init__(self, video_path: str, num_threads: int = 1, ctx=None):
        self.video_path = video_path
        self.num_threads = num_threads
        
        if DECORD_AVAILABLE:
            try:
                if ctx is None:
                    ctx = decord.cpu(0)
                self.reader = decord.VideoReader(video_path, ctx=ctx, num_threads=num_threads)
                self.use_decord = True
            except Exception as e:
                logger.warning(f"Failed to use decord, falling back to opencv: {e}")
                self.use_decord = False
                self._init_opencv()
        else:
            self.use_decord = False
            self._init_opencv()
    
    def _init_opencv(self):
        """Initialize opencv video reader as fallback."""
        import cv2
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def __len__(self):
        if self.use_decord:
            return len(self.reader)
        return self.frame_count
    
    def __getitem__(self, index):
        if self.use_decord:
            return self.reader[index].asnumpy()
        else:
            return self._get_frame_opencv(index)
    
    def _get_frame_opencv(self, index: int) -> np.ndarray:
        """Get frame using opencv."""
        import cv2
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            raise IndexError(f"Could not read frame {index}")
        # Convert BGR to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def get_batch(self, indices):
        """Get multiple frames."""
        if self.use_decord:
            return self.reader.get_batch(indices).asnumpy()
        else:
            frames = []
            for idx in indices:
                frames.append(self._get_frame_opencv(idx))
            return np.array(frames)
    
    def get_avg_fps(self) -> float:
        """Get average FPS."""
        if self.use_decord:
            return self.reader.get_avg_fps()
        return self.fps
    
    def __del__(self):
        if not self.use_decord and hasattr(self, 'cap'):
            self.cap.release()


def cpu(device_id: int = 0):
    """Compatibility function for decord.cpu()."""
    if DECORD_AVAILABLE:
        return decord.cpu(device_id)
    return None  # Not needed for opencv

